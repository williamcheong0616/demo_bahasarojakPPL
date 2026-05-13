#!/usr/bin/env python3
"""
Merges the SLM LoRA adapter into the base model and converts the result to
MLX format so it can be loaded by mlx_lm on Apple Silicon.

Run this ONCE on a CUDA machine (it needs bitsandbytes + GPU to load the PEFT model).
Copy the output directory to your MacBook and update SLM_BASE_MODEL in main.py to
point at it.

Usage:
  python convert_slm_to_mlx.py
  python convert_slm_to_mlx.py --base_model aisingapore/Llama-SEA-LION-v3.5-8B-R \
                                --adapter_path ./output/llama-8b/final \
                                --merged_path  ./slm-merged \
                                --mlx_path     ./slm-mlx
"""

import argparse
import subprocess
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL   = "aisingapore/Llama-SEA-LION-v3.5-8B-R"
ADAPTER_PATH = "./output/llama-8b/final"
MERGED_PATH  = "./slm-merged"
MLX_PATH     = "./slm-mlx"


def merge_peft(base_model_id: str, adapter_path: str, merged_path: str):
    print(f"Loading base model: {base_model_id}")
    # Load in fp16 — NOT 4-bit. bitsandbytes-quantized weights can't be saved to
    # safetensors; we need the full-precision weights to merge, then mlx_lm.convert
    # handles requantisation for the Mac side.
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"Attaching LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)

    print("Merging and unloading adapter...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {merged_path} ...")
    model.save_pretrained(merged_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_path)
    print("Merge complete.")


def convert_to_mlx(merged_path: str, mlx_path: str):
    print(f"Converting {merged_path} → {mlx_path} via mlx_lm.convert ...")
    result = subprocess.run(
        [
            sys.executable, "-m", "mlx_lm.convert",
            "--hf-path", merged_path,
            "--mlx-path", mlx_path,
            "--quantize",          # 4-bit quantisation for smaller footprint on Mac
            "--q-bits", "4",
        ],
        check=True,
    )
    print("Conversion complete.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model",   default=BASE_MODEL)
    parser.add_argument("--adapter_path", default=ADAPTER_PATH)
    parser.add_argument("--merged_path",  default=MERGED_PATH)
    parser.add_argument("--mlx_path",     default=MLX_PATH)
    parser.add_argument("--skip-merge",   action="store_true",
                        help="Skip PEFT merge step (use if merged_path already exists)")
    args = parser.parse_args()

    if not args.skip_merge:
        merge_peft(args.base_model, args.adapter_path, args.merged_path)

    convert_to_mlx(args.merged_path, args.mlx_path)

    print(f"\nDone. Copy '{args.mlx_path}' to your MacBook, then set:")
    print(f"  SLM_BASE_MODEL = '{args.mlx_path}'  (in main.py)")
