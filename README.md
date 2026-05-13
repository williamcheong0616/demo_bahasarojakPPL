# Bahasa Rojak AI Demo

A live demo of an end-to-end **ASR → SLM → TTS** pipeline for Bahasa Rojak (Malay/English code-switching). Speak a question, the system transcribes it, generates an answer with a fine-tuned Llama model, and reads the answer back.

---

## Requirements

| Dependency | macOS (Apple Silicon) | Linux / CUDA GPU |
|---|---|---|
| Python | 3.10 – 3.12 | 3.10 – 3.12 |
| ffmpeg | `brew install ffmpeg` | `sudo apt install ffmpeg libsndfile1` |
| CUDA | Not needed | 12.1+ recommended |
| RAM / VRAM | 16 GB unified RAM | 16 GB VRAM (SLM) |

---

## Setup

### macOS (Apple Silicon — MLX)

```bash
# 1. Install system dependency
brew install ffmpeg

# 2. Install Python packages
pip install -r requirements.txt -r requirements-mlx.txt

# 3. Unzip the Whisper LoRA adapter (if not already done)
# The final_adapter/ folder must exist in the project root.
# On the MLX path the adapter is not used — mlx_whisper uses the base turbo model.

# 4. Start the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Linux / CUDA GPU

```bash
# 1. Install system dependencies
sudo apt install ffmpeg libsndfile1

# 2. Install Python packages (no MLX)
pip install -r requirements.txt

# 3. Make sure your model files are present:
#    ./final_adapter/          — Whisper LoRA adapter
#    ./output/llama-8b/final/  — Llama-SEA-LION LoRA adapter

# 4. Start the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## First run

Startup takes **2–5 minutes** because models are downloaded and loaded sequentially before the first request is served. Watch the terminal for these lines:

```
[backend] MLX (Apple Silicon)          ← or PyTorch/CUDA
[startup] ASR ready ...
[startup] SLM ready.
[startup] TTS ready.
INFO:     Application startup complete.
```

If a model fails to load (e.g. not enough RAM for the SLM) you will see a `WARNING` line instead. The rest of the app still works — affected endpoints return `503`.

Then open **http://localhost:8000** in your browser.

---

## Using the demo

### Step 1 — Start

Click **"Ya, Mula!"** on the start screen. The browser will ask for microphone permission — allow it. The system plays a Bahasa Rojak greeting and shows the conversation screen.

### Step 2 — Ask a question (push-to-talk)

**Hold** the round button and speak. The button turns red and pulses while recording.  
**Release** to send. Three things happen automatically:

| Stage | What you see | What runs |
|---|---|---|
| Transcribing | Spinner: *Mengenal pasti audio...* | `/api/asr` — Whisper |
| Generating | Spinner: *Menjana respons...* | `/api/slm` — Llama-SEA-LION |
| Speaking | Spinner: *Menjana audio...* → audio plays | `/api/tts` — Scicom TTS |

Your words appear on the right (blue); the AI reply appears on the left (purple), then the answer is spoken aloud.

### Step 3 — Continue

Once audio finishes playing the button re-enables. Hold again to ask another question. The conversation history scrolls automatically.

---

## API reference

All endpoints are JSON in / JSON or audio out. You can test them directly with curl.

### `POST /api/greet`
Returns the greeting text and a base64-encoded WAV audio clip.

```bash
curl -X POST http://localhost:8000/api/greet
```
```json
{
  "text": "Halo! Selamat datang ke demo Bahasa Rojak AI...",
  "audio_b64": "<base64 WAV>"
}
```

### `POST /api/asr`
Accepts a multipart audio file, returns the transcript.

```bash
curl -X POST http://localhost:8000/api/asr \
  -F "file=@my_recording.wav"
```
```json
{ "transcript": "Apa khabar, boleh tolong saya?" }
```

Supported formats: `.wav`, `.webm`, `.mp3`, `.ogg`, `.mp4` — anything ffmpeg handles.

### `POST /api/slm`
Sends a text prompt to the language model, returns the generated reply.

```bash
curl -X POST http://localhost:8000/api/slm \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Apa itu machine learning?"}'
```
```json
{ "response": "Machine learning is a subset of AI yang..." }
```

### `POST /api/tts`
Converts text to speech, returns raw WAV bytes.

```bash
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Halo dunia!"}' \
  --output reply.wav
```

---

## SLM backends

The app supports three interchangeable SLM backends selected by the `SLM_BACKEND` environment variable:

| `SLM_BACKEND` | How it works | RAM used by app | Best for |
|---|---|---|---|
| `ollama` | HTTP calls to local Ollama server | ~0 MB | MacBook demo, low RAM |
| `mlx` | Loads model via `mlx_lm` into unified RAM | ~2–8 GB | Apple Silicon, no Ollama |
| `cuda` | Loads 4-bit quantised model via bitsandbytes | ~6 GB VRAM | NVIDIA GPU server |
| `auto` *(default)* | `mlx` on Apple Silicon, `cuda` otherwise | — | General |

### Using Ollama (recommended for MacBook demo)

Ollama runs the model in its own process — the FastAPI app uses almost no RAM for the SLM.

```bash
# 1. Install Ollama
#    Download from https://ollama.com  or:
brew install ollama        # macOS

# 2. Pull the model (downloads ~1.6 GB)
ollama pull gemma3:4b

# 3. Start the server with Ollama SLM
SLM_BACKEND=ollama uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Ollama starts automatically when you run `ollama pull` or `ollama run`. If you need to start it manually: `ollama serve`.

To use a different model, override `OLLAMA_MODEL`:

```bash
OLLAMA_MODEL=gemma3:4b SLM_BACKEND=ollama uvicorn main:app --reload --port 8000
```

Any model listed by `ollama list` works — e.g. `llama3.2:3b`, `phi4-mini`, `mistral:7b`.

## Model details

| Component | Ollama | macOS (MLX) | Linux (CUDA) |
|---|---|---|---|
| **ASR** | `mlx-community/whisper-large-v3-turbo` (Mac) or Whisper-LoRA (CUDA) | same | `openai/whisper-large-v3` + `./final_adapter/` LoRA |
| **SLM** | `gemma3:4b` via Ollama HTTP | `aisingapore/Llama-SEA-LION-v3.5-8B-R` via `mlx_lm` | Same + QLoRA `./output/llama-8b/final/` in 4-bit NF4 |
| **TTS** | `Scicom-intl/Multilingual-TTS-1.7B-Base` on MPS/CPU | same | same, on GPU |

> **Note (MLX/Ollama):** The fine-tuned LoRA adapters are in PyTorch/PEFT format and cannot be used on MLX or Ollama directly. The demo runs with the respective base models. To use your fine-tuned SLM on Mac, run `convert_slm_to_mlx.py` on a GPU machine first (see below).

---

## Using the fine-tuned SLM on Mac (optional)

Run this **once** on a CUDA machine (needs ~16 GB VRAM in fp16):

```bash
python convert_slm_to_mlx.py
# Outputs: ./slm-merged/  (full fp16 merged model)
#          ./slm-mlx/     (4-bit MLX model ready for Mac)
```

Copy `./slm-mlx/` to your MacBook, then update `main.py`:

```python
SLM_BASE_MODEL = "./slm-mlx"   # line ~80
```

Restart the server — it will now load your fine-tuned weights via `mlx_lm`.

Similarly for Whisper: run `merge_whisper.py` on a GPU machine to produce `./whisper-merged/`, then use it as the base in a custom `mlx_lm.convert` pipeline.

---

## Configuration

All tuneable constants are at the top of `main.py`:

```python
GREETING_TEXT    # The Bahasa Rojak greeting spoken at startup
ASR_BASE_MODEL   # HuggingFace model ID for Whisper (CUDA path)
ASR_ADAPTER_PATH # Path to Whisper LoRA adapter (CUDA path)
ASR_MLX_MODEL    # HuggingFace repo for mlx_whisper (Mac path)
SLM_BASE_MODEL   # HuggingFace model ID or local path for the LLM
SLM_ADAPTER_PATH # Path to Llama LoRA adapter (CUDA path)
TTS_MODEL_ID     # HuggingFace model ID for TTS
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `/api/slm` returns 503 | SLM failed to load (OOM or missing adapter) | Check terminal for `WARNING: SLM failed`. Ensure `./output/llama-8b/final/` exists (CUDA) or enough RAM is free (MLX). |
| `/api/asr` returns 503 | Whisper failed to load | CUDA: check `./final_adapter/` exists. MLX: `mlx_whisper` should self-recover on next request. |
| No audio plays | TTS returned empty or browser blocked autoplay | Check browser console. Safari may block audio without a user gesture — this is handled, but some ad-blockers interfere. |
| `bitsandbytes` error on Mac | Wrong requirements installed | Use `requirements-mlx.txt` on Mac, not the base `requirements.txt` alone. Do not install `bitsandbytes` on Apple Silicon. |
| "Cannot reach Ollama" at startup | Ollama not running | Run `ollama serve` in a separate terminal, or just run `ollama pull gemma3:4b` once (it auto-starts the server). |
| "Model not found in Ollama" | Model not pulled yet | Run `ollama pull gemma3:4b` (or whichever `OLLAMA_MODEL` you set). |
| Whisper transcribes garbage | Wrong language or noisy mic | Speak clearly. The model defaults to `language="ms"`. For English-only speech you can change this to `"en"` in `main.py` (`_mlx_transcribe`) or `whisper_lora_inference.py.py`. |
| `mlx` not found after install | Installed on Intel Mac or wrong Python | MLX only runs on Apple Silicon (M1/M2/M3/M4). Check with `python -c "import mlx.core; print('ok')"`. |
