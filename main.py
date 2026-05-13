#!/usr/bin/env python3
"""
Bahasa Rojak AI Demo — FastAPI backend.

Endpoints:
  POST /api/greet  → {text, audio_b64}
  POST /api/asr    → multipart audio → {transcript}
  POST /api/slm    → {prompt} → {response}
  POST /api/tts    → {text} → audio/wav bytes

Run:
  python run.py                    # recommended (auto-detects backend)
  uvicorn main:app --port 8000     # manual

SLM backend is selected via the SLM_BACKEND env var:
  ollama  — local Ollama server (lowest RAM, good for MacBook demo)
  mlx     — mlx_lm on Apple Silicon
  cuda    — PyTorch + bitsandbytes 4-bit on NVIDIA GPU
  auto    — mlx if Apple Silicon, cuda otherwise (default)
"""

import asyncio
import base64
import gc
import io
import os
import re
import tempfile
from contextlib import asynccontextmanager

import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM

from asr import load_model as _load_asr, transcribe as _asr_transcribe

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GREETING_TEXT = (
    "Halo! Selamat datang ke demo Bahasa Rojak AI. "
    "Saya siap nak bantu you. Tekan butang untuk mula!"
)

# ASR — prefers local merged weights; falls back to HuggingFace download
ASR_MODEL_PATH = os.environ.get("ASR_MODEL_PATH", "./whisper-merged")
ASR_BASE_MODEL = "openai/whisper-large-v3"

# ASR (MLX path — separate, uses mlx_whisper)
ASR_MLX_MODEL  = "mlx-community/whisper-large-v3-turbo"

# SLM
SLM_BASE_MODEL   = "aisingapore/Llama-SEA-LION-v3.5-8B-R"
SLM_ADAPTER_PATH = "./output/llama-8b/final"
OLLAMA_MODEL     = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

# TTS
TTS_MODEL_ID  = "Scicom-intl/Multilingual-TTS-1.7B-Base"
TTS_CODEC_ID  = "neuphonic/neucodec"
TTS_SPEAKER   = "husein"   # default speaker voice
TTS_SAMPLE_RATE = 24000

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

# ASR: MLX Whisper on Apple Silicon, otherwise use asr.py (CUDA / CPU)
try:
    import mlx.core as _mx  # noqa: F401
    USE_MLX = True
except ImportError:
    USE_MLX = False

# SLM backend
_slm_env = os.environ.get("SLM_BACKEND", "auto").lower()
if _slm_env == "ollama":
    SLM_BACKEND = "ollama"
elif _slm_env == "mlx":
    SLM_BACKEND = "mlx"
elif _slm_env == "cuda":
    SLM_BACKEND = "cuda"
else:
    SLM_BACKEND = "mlx" if USE_MLX else "cuda"

print(f"[backend] ASR={'mlx_whisper' if USE_MLX else 'Whisper (transformers)'}  "
      f"SLM={SLM_BACKEND}  TTS=NeuCodec+Qwen3")

# ---------------------------------------------------------------------------
# Backend-specific SLM imports
# ---------------------------------------------------------------------------

if SLM_BACKEND == "ollama":
    from inference_ollama import generate as _slm_generate, load_model as _slm_load
elif SLM_BACKEND == "mlx":
    from inference_mlx import generate as _slm_generate, load_model as _slm_load
else:
    from inference import generate as _slm_generate, load_model as _slm_load

if USE_MLX:
    import mlx_whisper

# ---------------------------------------------------------------------------
# Model singletons
# ---------------------------------------------------------------------------

_asr_pipe      = None   # transformers ASR pipeline (CUDA) or model-repo str (MLX)
_asr_bad_words = None   # Latin-only suppression list (CUDA path only)
_slm_model     = None
_slm_tokenizer = None
_tts_model     = None
_tts_tokenizer = None
_tts_codec     = None
_tts_device    = "cpu"

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _asr_pipe, _asr_bad_words, _slm_model, _slm_tokenizer
    global _tts_model, _tts_tokenizer, _tts_codec, _tts_device

    # ── ASR ──────────────────────────────────────────────────────────────────
    if USE_MLX:
        print(f"[startup] ASR: mlx_whisper ({ASR_MLX_MODEL})")
        _asr_pipe = ASR_MLX_MODEL   # non-None sentinel; model downloads on first call
        print("[startup] ASR ready (downloads on first request if not cached).")
    else:
        try:
            _asr_pipe, _asr_bad_words = _load_asr(ASR_MODEL_PATH, ASR_BASE_MODEL)
            print("[startup] ASR ready.")
        except Exception as e:
            print(f"[startup] WARNING: ASR failed — {e}. /api/asr will return 503.")

    # ── SLM ──────────────────────────────────────────────────────────────────
    _slm_label = OLLAMA_MODEL if SLM_BACKEND == "ollama" else SLM_BASE_MODEL
    print(f"[startup] SLM ({SLM_BACKEND}): {_slm_label}")
    try:
        _slm_id = OLLAMA_MODEL if SLM_BACKEND == "ollama" else SLM_BASE_MODEL
        _slm_model, _slm_tokenizer = _slm_load(SLM_ADAPTER_PATH, _slm_id)
        print("[startup] SLM ready.")
    except Exception as e:
        print(f"[startup] WARNING: SLM failed — {e}. /api/slm will return 503.")

    # ── TTS ──────────────────────────────────────────────────────────────────
    print(f"[startup] TTS: {TTS_MODEL_ID}  codec: {TTS_CODEC_ID}")
    try:
        from neucodec import NeuCodec
        _tts_device = (
            "mps" if (USE_MLX and torch.backends.mps.is_available()) else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
        _tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_ID)
        _tts_model = AutoModelForCausalLM.from_pretrained(
            TTS_MODEL_ID,
            torch_dtype=torch.bfloat16 if _tts_device != "cpu" else torch.float32,
            device_map=_tts_device,
        )
        _tts_model.eval()
        _tts_codec = NeuCodec.from_pretrained(TTS_CODEC_ID).eval().to(_tts_device)
        print(f"[startup] TTS ready (device={_tts_device}).")
    except Exception as e:
        print(f"[startup] WARNING: TTS failed — {e}. /api/tts will return 503.")

    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Bahasa Rojak AI Demo", lifespan=lifespan)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class GreetResponse(BaseModel):
    text: str
    audio_b64: str

class ASRResponse(BaseModel):
    transcript: str

class SLMRequest(BaseModel):
    prompt: str

class SLMResponse(BaseModel):
    response: str

class TTSRequest(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Memory management
# ---------------------------------------------------------------------------

def _clear_memory():
    """Release GPU/CPU cache after each inference so memory doesn't accumulate."""
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass  # CUDA context may be in error state; don't let cleanup crash the endpoint


# ---------------------------------------------------------------------------
# TTS helpers
# ---------------------------------------------------------------------------

def _text_to_wav(text: str, speaker: str = TTS_SPEAKER) -> bytes:
    if _tts_model is None or _tts_codec is None:
        raise RuntimeError("TTS not loaded")

    prompt = f"<|im_start|>{speaker}: {text}<|speech_start|>"
    inputs = _tts_tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(_tts_device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _tts_model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.8,
            repetition_penalty=1.15,
        )

    generated = _tts_tokenizer.decode(outputs[0], skip_special_tokens=False)
    speech_part = generated.split("<|speech_start|>")[-1]
    audio_tokens = [int(t) for t in re.findall(r"<\|s_(\d+)\|>", speech_part)]

    if not audio_tokens:
        raise RuntimeError("TTS generated no audio tokens — check speaker name or prompt format")

    audio_codes = torch.tensor(audio_tokens, dtype=torch.long)[None, None].to(_tts_device)
    with torch.no_grad():
        waveform = _tts_codec.decode_code(audio_codes)

    audio_np = waveform[0, 0].cpu().float().numpy()
    buf = io.BytesIO()
    sf.write(buf, audio_np, TTS_SAMPLE_RATE, format="WAV")
    buf.seek(0)
    return buf.read()


def _tts_to_b64(text: str) -> str:
    return base64.b64encode(_text_to_wav(text)).decode()


# ---------------------------------------------------------------------------
# ASR helpers
# ---------------------------------------------------------------------------

def _mlx_transcribe(audio_path: str, model_repo: str) -> str:
    result = mlx_whisper.transcribe(audio_path, path_or_hf_repo=model_repo, language="ms")
    return result["text"].strip()


def _write_tmp(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        return f.name


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/greet", response_model=GreetResponse)
async def greet():
    import traceback
    audio_b64 = ""
    if _tts_model is None or _tts_codec is None:
        print("[greet] TTS not loaded — skipping audio")
    else:
        try:
            audio_b64 = await asyncio.to_thread(_tts_to_b64, GREETING_TEXT)
        except Exception:
            print(f"[greet] TTS error:\n{traceback.format_exc()}")
        finally:
            _clear_memory()
    return GreetResponse(text=GREETING_TEXT, audio_b64=audio_b64)


@app.post("/api/asr", response_model=ASRResponse)
async def asr(file: UploadFile = File(...)):
    if _asr_pipe is None:
        raise HTTPException(status_code=503, detail="ASR model not loaded")

    audio_bytes = await file.read()
    suffix = os.path.splitext(file.filename or ".webm")[1] or ".webm"
    tmp_path = _write_tmp(audio_bytes, suffix)
    try:
        if USE_MLX:
            text = await asyncio.to_thread(_mlx_transcribe, tmp_path, _asr_pipe)
        else:
            text = await asyncio.to_thread(_asr_transcribe, tmp_path, _asr_pipe, _asr_bad_words)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        _clear_memory()

    return ASRResponse(transcript=text)


@app.post("/api/slm", response_model=SLMResponse)
async def slm(body: SLMRequest):
    if _slm_model is None or _slm_tokenizer is None:
        raise HTTPException(status_code=503, detail="SLM model not available")

    try:
        text = await asyncio.to_thread(_slm_generate, _slm_model, _slm_tokenizer, body.prompt)
    finally:
        _clear_memory()
    return SLMResponse(response=text)


@app.post("/api/tts")
async def tts(body: TTSRequest):
    if _tts_model is None or _tts_codec is None:
        raise HTTPException(status_code=503, detail="TTS model not available")
    try:
        wav_bytes = await asyncio.to_thread(_text_to_wav, body.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        _clear_memory()
    return Response(content=wav_bytes, media_type="audio/wav")
