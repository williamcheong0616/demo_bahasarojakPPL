# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A demo webapp showing an end-to-end **ASR → SLM → TTS** pipeline for Bahasa Rojak (Malay/English code-switching). A user clicks a button, speaks via push-to-talk, and the system transcribes, generates a response via a fine-tuned LLM, and speaks the answer back.

## How to run

### Apple Silicon Mac (MLX)
```bash
brew install ffmpeg
pip install -r requirements.txt -r requirements-mlx.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### CUDA GPU (Linux / Windows)
```bash
sudo apt install ffmpeg libsndfile1
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`. Startup takes 2–5 minutes — all three models load sequentially. Watch stdout for `[startup] ... ready.` confirmations.

The backend is selected automatically: if `mlx` is importable the app uses MLX; otherwise it uses PyTorch/CUDA.

## Architecture

```
main.py           FastAPI backend — 4 endpoints + static file serving
inference.py      SLM wrapper (do not modify) — load_model() + generate()
static/index.html Single-page frontend (self-contained, no build step)
requirements.txt
```

### Endpoints

| Path | Input | Output | Model |
|---|---|---|---|
| `GET /api/status` | — | `{asr, slm, tts}` booleans | — |
| `POST /api/greet` | — | `{text, audio_b64}` | TTS |
| `POST /api/asr` | multipart audio file | `{transcript}` | Whisper |
| `POST /api/slm` | `{prompt, history?}` | `{response}` | Llama-SEA-LION |
| `POST /api/tts` | `{text}` | `audio/wav` bytes | Scicom TTS |

`/api/status` reports which models loaded successfully — load success, not ongoing
health. The frontend uses it to show a status dot and to disable push-to-talk when ASR
is unavailable.

### Conversation memory

`/api/slm` accepts an optional `history` of `{role, content}` messages (roles `user` /
`assistant`) that is prepended to the prompt as prior turns. The client trims to the
last 10 messages — 5 exchanges — and the server re-caps at `MAX_HISTORY` in `main.py`,
which is authoritative.

All three `generate()` backends take a keyword-only `history=None`. `inference_ollama.py`
also sets `num_ctx: 8192`; the Ollama default of 2048 would silently truncate the oldest
turns given the ~400-token system prompt.

### Model loading

All three models are globals loaded once at startup inside the FastAPI `lifespan` context. If a model fails to load (e.g. GPU OOM for the SLM), the global stays `None` and the affected endpoint returns `503` — the rest of the app still works.

- **Whisper** — `openai/whisper-large-v3` base + local LoRA adapter at `./final_adapter`, loaded via `whisper_lora_inference.py.py`. Builds a Latin-only `bad_words_ids` filter at startup. The module is imported at runtime via `importlib` because of the `.py.py` filename.
- **SLM** — `aisingapore/Llama-SEA-LION-v3.5-8B-R` base + local QLoRA adapter at `./output/llama-8b/final`, 4-bit NF4 quantization
- **TTS** — `Scicom-intl/Multilingual-TTS-1.7B-Base` via HuggingFace `pipeline("text-to-speech")`

### Key implementation notes

- Whisper requires a file path (not bytes). The ASR endpoint writes the upload to a `NamedTemporaryFile`, transcribes, then deletes it in `finally`.
- All model inference is wrapped in `asyncio.to_thread()` to avoid blocking the async event loop.
- TTS pipeline output key is handled with `out.get("audio", out.get("waveform"))` to accommodate model variants.
- The frontend uses `MediaRecorder` for audio capture; MIME type is detected at runtime (`webm`, `ogg`, fallback). Push-to-talk uses pointer events, so touch works and nothing depends on hover.
- No CORS config needed — the frontend is served by the same FastAPI process.
- Typed input skips ASR and runs SLM → TTS through the same `runTurn()` path as voice, so the app stays usable when the mic is blocked or ASR failed to load.

### Interface language

The header language selector (Malay / English / Chinese) translates **the interface
only** — it is chrome. It does not change the AI's reply language, the Whisper hint
(`language="ms"`), or `GREETING_TEXT`. The project's identity is Bahasa Rojak; the AI's
own voice stays fixed while the surrounding UI adapts.

Strings live in the `I18N` object in `static/index.html`. Elements carry `data-i18n`
attributes and `applyLang()` rewrites them. The choice persists in `localStorage`. When
adding interface text, add a key to all three dictionaries.

## Configuration constants (top of `main.py`)

```python
GREETING_TEXT    # Bahasa Rojak greeting spoken at startup
ASR_BASE_MODEL   # HuggingFace model ID for Whisper base (default: openai/whisper-large-v3)
ASR_ADAPTER_PATH # Path to Whisper LoRA adapter (default: ./final_adapter)
SLM_ADAPTER_PATH # Path to QLoRA adapter weights (default: ./output/llama-8b/final)
SLM_BASE_MODEL   # HuggingFace model ID for the base LLM
TTS_MODEL_ID     # HuggingFace model ID for TTS
```

## Utility scripts

| Script | Purpose | Runs on |
|---|---|---|
| `merge_whisper.py` | Merge Whisper base + `./final_adapter` LoRA → `./whisper-merged/` | CUDA GPU |
| `convert_slm_to_mlx.py` | Merge SLM PEFT adapter → convert to MLX 4-bit → `./slm-mlx/` | CUDA GPU |

To use the fine-tuned SLM on a MacBook, run `convert_slm_to_mlx.py` on a GPU machine, copy `./slm-mlx/` to the Mac, then set `SLM_BASE_MODEL = "./slm-mlx"` in `main.py`.

## MLX backend notes

- **ASR**: Uses `mlx-community/whisper-large-v3-mlx` (base model, no LoRA). The PyTorch LoRA adapter (`./final_adapter`) cannot be loaded by `mlx_whisper` directly.
- **SLM**: Uses `mlx_lm.load()`. Without conversion, the base model is loaded without the LoRA adapter.
- **TTS**: Runs via PyTorch on MPS (Apple GPU) or CPU — no MLX TTS package needed.
