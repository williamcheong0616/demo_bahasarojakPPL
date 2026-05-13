#!/usr/bin/env python3
"""
Merges the Whisper-large-v3 base model with the LoRA adapter in ./final_adapter/
and saves the result as a standalone model ready for direct use (no PEFT required).

Usage:
  python merge_whisper.py
  python merge_whisper.py --base_model openai/whisper-large-v3 \
                          --lora_path ./final_adapter \
                          --output_path ./whisper-merged
"""

import argparse
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

BASE_MODEL_ID = "openai/whisper-large-v3"
LORA_PATH     = "./final_adapter"
OUTPUT_PATH   = "./whisper-merged"


def merge(base_model_id: str, lora_path: str, output_path: str):
    print(f"Loading processor from {base_model_id}...")
    processor = WhisperProcessor.from_pretrained(base_model_id)

    print(f"Loading base model from {base_model_id}...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )

    print(f"Attaching LoRA adapter from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)

    print("Merging weights and unloading adapter...")
    model = model.merge_and_unload()
    model.eval()

    print(f"Saving merged model to {output_path}...")
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)

    print(f"\nDone. Merged model saved to: {output_path}")
    print("You can now load it with:")
    print(f'  WhisperForConditionalGeneration.from_pretrained("{output_path}")')
    print(f'  WhisperProcessor.from_pretrained("{output_path}")')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",  default=BASE_MODEL_ID)
    parser.add_argument("--lora_path",   default=LORA_PATH)
    parser.add_argument("--output_path", default=OUTPUT_PATH)
    args = parser.parse_args()

    merge(args.base_model, args.lora_path, args.output_path)
