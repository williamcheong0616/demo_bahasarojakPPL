import os
import re
import argparse
import torch
import librosa
import numpy as np
import warnings
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from peft import PeftModel

def setup_inference_pipeline(base_model_id, checkpoint_path):
    """
    Loads the processor, builds the bad words list, and loads the merged LoRA model.
    """
    print("Loading Processor...")
    processor = WhisperProcessor.from_pretrained(base_model_id)

    print("Building 'Latin-Only' Wall...")
    allowed_pattern = re.compile(r'^[a-zA-Z0-9\s.,\'?! \-_]+$')
    bad_words_ids = []

    for token, idx in processor.tokenizer.get_vocab().items():
        clean_token = token.replace('Ġ', ' ').replace(' ', ' ')
        if token.startswith('<|'):
            continue
        if not allowed_pattern.match(clean_token):
            bad_words_ids.append([idx])
            
    print(f"Banned {len(bad_words_ids)} non-Latin tokens.")

    print(f"Loading Base Model ({base_model_id})...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float16, 
        device_map="cuda:0"
    )

    print(f"Attaching and Merging LoRA Adapter from {checkpoint_path}...")
    model_tuned = PeftModel.from_pretrained(base_model, checkpoint_path)
    model_tuned = model_tuned.merge_and_unload() # Merge weights for faster inference

    print("Building Pipeline...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_tuned,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16
    )
    
    return pipe, bad_words_ids

def transcribe_audio(audio_path, pipe, bad_words_ids, language="ms"):
    """
    Takes an audio file path and runs it through the loaded pipeline.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load and resample audio to 16kHz
    audio_array, sr = librosa.load(audio_path, sr=16000, mono=True)
    audio_array = np.asarray(audio_array, dtype=np.float32).squeeze()

    gen_kwargs = {
        "task": "transcribe", 
        "language": language, 
        "bad_words_ids": bad_words_ids
    }

    # Run Inference
    result = pipe(
        {"raw": audio_array, "sampling_rate": sr},
        generate_kwargs=gen_kwargs
    )
    
    return result["text"].strip()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an audio file using Fine-tuned Whisper.")
    parser.add_argument("--audio", type=str, required=True, help="Path to the audio file (.wav, .mp3, etc.)")
    parser.add_argument("--base_model", type=str, default="openai/whisper-large-v3", help="Base model ID")
    parser.add_argument("--lora_path", type=str, default="./whisper-large-v3-malay-lora-aug-q/final_adapter", help="Path to LoRA weights")
    
    args = parser.parse_args()

    # 1. Initialize once
    pipe, bad_words_ids = setup_inference_pipeline(args.base_model, args.lora_path)

    # 2. Transcribe
    print(f"\nTranscribing: {args.audio}")
    transcription = transcribe_audio(args.audio, pipe, bad_words_ids, language="ms")
    
    # 3. Output
    print("\n" + "="*50)
    print("TRANSCRIPTION RESULT:")
    print("="*50)
    print(transcription)
    print("="*50 + "\n")


