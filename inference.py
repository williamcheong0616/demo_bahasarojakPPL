#!/usr/bin/env python3
"""
Inference with QLoRA fine-tuned Llama-SEA-LION-v3.5-8B-R.

Usage:
  python llama/inference.py --prompt "Macam mana AI boleh bantu cikgu?"
  python llama/inference.py --interactive
"""

import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

BASE_MODEL_ID = "aisingapore/Llama-SEA-LION-v3.5-8B-R"
ADAPTER_PATH = "./output/llama-8b/final"


def load_model(adapter_path, base_model_id):
    print(f"🔄 Loading base model: {base_model_id}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id, quantization_config=bnb_config, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    adapter_cfg = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(adapter_cfg):
        print(f"🔗 Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print(f"⚠️  Adapter not found at '{adapter_path}' — using base model only")
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Ready!\n")
    return model, tokenizer


_SYSTEM_PROMPT = """\
You are a Malaysian AI assistant who ALWAYS responds in Bahasa Rojak — compulsory code-switching between Malay and English in every single sentence. This is non-negotiable.

RULES (strictly follow):
1. Every sentence MUST contain BOTH Malay and English words mixed together. Never write a sentence that is 100% Malay or 100% English.
2. Use Malaysian filler words: lah, leh, lor, kan, mah, weh, wei, bro, sis.
3. English technical terms, nouns, and verbs can be left in English inside a Malay sentence structure.
4. Keep it casual and friendly, like texting a Malaysian friend.

EXAMPLES of correct Bahasa Rojak replies:
- "Okay lah, you boleh try restart your computer dulu, then check balik the settings."
- "Actually cara terbaik is to save your work first, lepas tu baru close the app."
- "Weh, AI ni memang best untuk automate boring tasks, senang je bro."
- "Kalau you nak learn programming, start with Python dulu lah, senang nak faham."
- "So basically, machine learning tu pakai data untuk train a model, then the model will predict new results."
- "Eh, don't worry lah — just follow these steps and everything will be fine one."

NEVER respond in pure Malay or pure English. Always mix both languages in every sentence.\
"""


def generate(model, tokenizer, instruction, input_ctx="",
             max_new_tokens=512, temperature=0.7):
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": instruction + (f"\n\n{input_ctx}" if input_ctx else "")},
    ]
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, thinking_mode="off")
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=max(1e-5, temperature), top_p=0.9, top_k=50,
            repetition_penalty=1.1, do_sample=True,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def interactive(model, tokenizer):
    print("=" * 60)
    print("🤖 Llama-SEA-LION Interactive — type 'quit' to stop")
    print("=" * 60)
    while True:
        try:
            instr = input("\n📝 Instruction: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if instr.lower() in ("quit", "exit", "q"):
            break
        if not instr:
            continue
        ctx = input("📎 Context (Enter to skip): ").strip()
        print(f"\n🤖 {generate(model, tokenizer, instr, ctx)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", "-p", type=str)
    parser.add_argument("--input", "-i", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--adapter-path", default=ADAPTER_PATH)
    parser.add_argument("--base-model", default=BASE_MODEL_ID)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter_path, args.base_model)

    if args.interactive:
        interactive(model, tokenizer)
    elif args.prompt:
        print(f"🤖 {generate(model, tokenizer, args.prompt, args.input, args.max_tokens, args.temperature)}")
    else:
        print("Provide --prompt or --interactive")


if __name__ == "__main__":
    main()
