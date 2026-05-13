"""
Whisper ASR module for Bahasa Rojak demo.

Loads merged LoRA weights from a local directory if present, otherwise downloads
the base model from HuggingFace. Applies a Latin-only token filter so the model
outputs clean Malay/English romanised text.

Usage:
  from asr import load_model, transcribe
  pipe, bad_words = load_model("./whisper-merged", "openai/whisper-large-v3")
  text = transcribe("recording.webm", pipe, bad_words)
"""

import os
import re

import librosa
import numpy as np
import torch


def load_model(model_path: str, base_model_id: str = "openai/whisper-large-v3"):
    """
    Load Whisper pipeline.

    If model_path exists on disk (i.e. merged weights are present) it loads from
    there. Otherwise it downloads base_model_id from HuggingFace.

    Returns:
        pipe          — transformers ASR pipeline
        bad_words_ids — list of token-id lists to suppress non-Latin output
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from transformers import pipeline as _pipeline

    source = model_path if os.path.isdir(model_path) else base_model_id
    if source == model_path:
        print(f"[ASR] Loading merged Whisper weights from {model_path}")
    else:
        print(f"[ASR] Merged weights not found at '{model_path}' — "
              f"downloading {base_model_id} from HuggingFace")

    processor = WhisperProcessor.from_pretrained(source)

    # Build Latin-only suppression list — keeps a-z, A-Z, digits, basic punctuation.
    # This preserves Malay/English romanisation and blocks CJK / Devanagari / etc.
    _latin = re.compile(r"^[a-zA-Z0-9\s.,'\"\-_?!]+$")
    bad_words_ids = [
        [idx]
        for token, idx in processor.tokenizer.get_vocab().items()
        if not token.startswith("<|")
        and not _latin.match(
            token.replace("Ġ", " ").replace("▁", " ")  # BPE / SPM space markers
        )
    ]
    print(f"[ASR] Latin-only filter active — {len(bad_words_ids)} tokens suppressed.")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"[ASR] Loading model on {device} ({dtype}) ...")
    model = WhisperForConditionalGeneration.from_pretrained(
        source,
        torch_dtype=dtype,
        device_map=device,
    )

    pipe = _pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=dtype,
    )

    return pipe, bad_words_ids


def transcribe(audio_path: str, pipe, bad_words_ids,
               language: str = "ms") -> str:
    """
    Transcribe an audio file using the loaded Whisper pipeline.

    Resamples to 16 kHz via librosa (handles webm, mp4, ogg, wav, mp3, etc.
    as long as ffmpeg is installed).
    """
    audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio_array = np.asarray(audio_array, dtype=np.float32).squeeze()

    result = pipe(
        {"raw": audio_array, "sampling_rate": sr},
        generate_kwargs={
            "task": "transcribe",
            "language": language,
            "bad_words_ids": bad_words_ids,
        },
    )
    return result["text"].strip()
