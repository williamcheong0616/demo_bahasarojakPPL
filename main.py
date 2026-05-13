#!/usr/bin/env python3
"""
Bahasa Rojak AI Demo — FastAPI backend.

Endpoints:
  POST /api/greet  → {text, audio_b64}
  POST /api/asr    → multipart audio → {transcript}
  POST /api/slm    → {prompt} → {response}
  POST /api/tts    → {text} → audio/wav bytes

Run:
  uvicorn main:app --reload --host 0.0.0.0 --port 8000

Backend is selected automatically at startup:
  - Apple Silicon Mac (MLX available) → MLX path
  - CUDA GPU                          → PyTorch/PEFT/bitsandbytes path
"""

import asyncio
import base64
import importlib.util
import io
import os
import tempfile
from contextlib import asynccontextmanager

import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import transformers as _transformers

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GREETING_TEXT = (
    "Halo! Selamat datang ke demo Bahasa Rojak AI. "
    "Saya siap nak bantu you. Tekan butang untuk mula!"
)
# ASR
ASR_BASE_MODEL   = "openai/whisper-large-v3"
ASR_ADAPTER_PATH = "./final_adapter"
ASR_MLX_MODEL    = "mlx-community/whisper-large-v3-turbo"

# SLM — HuggingFace model for CUDA/MLX paths
SLM_BASE_MODEL   = "aisingapore/Llama-SEA-LION-v3.5-8B-R"
SLM_ADAPTER_PATH = "./output/llama-8b/final"

# Ollama — model tag to use when SLM_BACKEND=ollama
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")

# TTS
TTS_MODEL_ID = "Scicom-intl/Multilingual-TTS-1.7B-Base"

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

# ASR backend: MLX if Apple Silicon, otherwise CUDA/CPU
try:
    import mlx.core as _mx  # noqa: F401
    USE_MLX = True
except ImportError:
    USE_MLX = False

# SLM backend: set SLM_BACKEND env var to override.
#   "ollama" — use local Ollama server (lowest RAM, recommended for MacBook demo)
#   "mlx"    — use mlx_lm (Apple Silicon, loads model into unified RAM)
#   "cuda"   — use PyTorch + bitsandbytes (NVIDIA GPU)
#   "auto"   — cuda if available, else mlx if available, else cuda (will warn at load)
_slm_env = os.environ.get("SLM_BACKEND", "auto").lower()
if _slm_env == "ollama":
    SLM_BACKEND = "ollama"
elif _slm_env == "mlx":
    SLM_BACKEND = "mlx"
elif _slm_env == "cuda":
    SLM_BACKEND = "cuda"
else:  # auto
    SLM_BACKEND = "mlx" if USE_MLX else "cuda"

print(f"[backend] ASR={'mlx_whisper' if USE_MLX else 'Whisper-LoRA/CUDA'}  "
      f"SLM={SLM_BACKEND}  TTS=transformers")

# ---------------------------------------------------------------------------
# Backend-specific imports
# ---------------------------------------------------------------------------

# ASR
if USE_MLX:
    import mlx_whisper
else:
    _wli_spec = importlib.util.spec_from_file_location(
        "whisper_lora_inference", "./whisper_lora_inference.py.py"
    )
    _wli = importlib.util.module_from_spec(_wli_spec)
    _wli_spec.loader.exec_module(_wli)
    setup_asr_pipeline = _wli.setup_inference_pipeline
    transcribe_audio   = _wli.transcribe_audio

# SLM
if SLM_BACKEND == "ollama":
    from inference_ollama import generate, load_model
elif SLM_BACKEND == "mlx":
    from inference_mlx import generate, load_model
else:
    from inference import generate, load_model

# ---------------------------------------------------------------------------
# Model singletons
# ---------------------------------------------------------------------------

# ASR: on CUDA = HF pipeline + bad_words list; on MLX = model-repo string
_asr_pipe      = None   # also used as the mlx model path string on MLX
_asr_bad_words = None   # unused on MLX

_slm_model     = None
_slm_tokenizer = None
_tts_pipeline  = None

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _asr_pipe, _asr_bad_words, _slm_model, _slm_tokenizer, _tts_pipeline

    # ── ASR ──────────────────────────────────────────────────────────────────
    if USE_MLX:
        # mlx_whisper downloads/caches the model on first transcribe call;
        # store the repo string so the endpoint can pass it through.
        print(f"[startup] ASR backend: mlx_whisper ({ASR_MLX_MODEL})")
        _asr_pipe = ASR_MLX_MODEL          # sentinel: non-None means "ready"
        _asr_bad_words = None
        print("[startup] ASR ready (model will download on first request if not cached).")
    else:
        print(f"[startup] Loading Whisper LoRA ASR ({ASR_BASE_MODEL} + {ASR_ADAPTER_PATH})...")
        try:
            _asr_pipe, _asr_bad_words = setup_asr_pipeline(ASR_BASE_MODEL, ASR_ADAPTER_PATH)
            print("[startup] ASR ready.")
        except Exception as e:
            print(f"[startup] WARNING: ASR failed — {e}. /api/asr will return 503.")

    # ── SLM ──────────────────────────────────────────────────────────────────
    _slm_label = OLLAMA_MODEL if SLM_BACKEND == "ollama" else SLM_BASE_MODEL
    print(f"[startup] Loading SLM via {SLM_BACKEND} ({_slm_label})...")
    try:
        _slm_model_id = OLLAMA_MODEL if SLM_BACKEND == "ollama" else SLM_BASE_MODEL
        _slm_model, _slm_tokenizer = load_model(SLM_ADAPTER_PATH, _slm_model_id)
        print("[startup] SLM ready.")
    except Exception as e:
        print(f"[startup] WARNING: SLM failed — {e}. /api/slm will return 503.")

    # ── TTS ──────────────────────────────────────────────────────────────────
    print(f"[startup] Loading TTS ({TTS_MODEL_ID})...")
    try:
        if USE_MLX:
            # MPS can be unstable for some HF TTS models; CPU is safe on Mac
            tts_device = "mps" if torch.backends.mps.is_available() else "cpu"
            _tts_pipeline = _transformers.pipeline("text-to-speech", model=TTS_MODEL_ID, device=tts_device)
        else:
            _tts_pipeline = _transformers.pipeline("text-to-speech", model=TTS_MODEL_ID)
        print("[startup] TTS ready.")
    except Exception as e:
        print(f"[startup] WARNING: TTS failed — {e}. /api/tts will return 503.")

    yield


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
# TTS helpers
# ---------------------------------------------------------------------------

def _text_to_wav(text: str) -> bytes:
    if _tts_pipeline is None:
        raise RuntimeError("TTS not loaded")
    out = _tts_pipeline(text)
    audio_np = out.get("audio", out.get("waveform"))
    if audio_np is None:
        raise RuntimeError(f"Unexpected TTS output keys: {list(out.keys())}")
    sr = out["sampling_rate"]
    if audio_np.ndim > 1:
        audio_np = audio_np.squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio_np, sr, format="WAV")
    buf.seek(0)
    return buf.read()


def _tts_to_b64(text: str) -> str:
    return base64.b64encode(_text_to_wav(text)).decode()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tmp(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        return f.name


def _mlx_transcribe(audio_path: str, model_repo: str) -> str:
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=model_repo,
        language="ms",
    )
    return result["text"].strip()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/greet", response_model=GreetResponse)
async def greet():
    try:
        audio_b64 = await asyncio.to_thread(_tts_to_b64, GREETING_TEXT)
    except Exception as e:
        print(f"[greet] TTS error: {e}")
        audio_b64 = ""
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
            text = await asyncio.to_thread(transcribe_audio, tmp_path, _asr_pipe, _asr_bad_words)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return ASRResponse(transcript=text)


@app.post("/api/slm", response_model=SLMResponse)
async def slm(body: SLMRequest):
    if _slm_model is None or _slm_tokenizer is None:
        raise HTTPException(status_code=503, detail="SLM model not available")

    text = await asyncio.to_thread(generate, _slm_model, _slm_tokenizer, body.prompt)
    return SLMResponse(response=text)


@app.post("/api/tts")
async def tts(body: TTSRequest):
    if _tts_pipeline is None:
        raise HTTPException(status_code=503, detail="TTS model not available")
    try:
        wav_bytes = await asyncio.to_thread(_text_to_wav, body.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return Response(content=wav_bytes, media_type="audio/wav")
