"""
MLX-based SLM inference for Apple Silicon.

Matches the interface of inference.py so main.py can swap them at runtime.
Note: the PEFT LoRA adapter (./output/llama-8b/final) is in PyTorch format and
cannot be loaded directly by mlx_lm. Run convert_to_mlx.py first to produce a
merged MLX model, then set SLM_BASE_MODEL to that local path.
"""

from mlx_lm import load, generate as _mlx_generate


def load_model(adapter_path: str, base_model_id: str):
    """Load SLM via mlx_lm. adapter_path is unused (see module docstring)."""
    print(f"[MLX] Loading SLM: {base_model_id}")
    model, tokenizer = load(base_model_id)
    print("[MLX] SLM ready.")
    return model, tokenizer


_SYSTEM_PROMPT = (
    "You are a helpful AI assistant that speaks Bahasa Rojak — "
    "the natural, casual mix of Malay and English used in everyday Malaysian conversation. "
    "Blend both languages naturally, like 'Okay lah, you boleh try this method...' or "
    "'Actually, cara terbaik is to...'. Keep answers concise and conversational."
)


def generate(model, tokenizer, instruction: str, input_ctx: str = "",
             max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": instruction + (f"\n\n{input_ctx}" if input_ctx else "")},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, thinking_mode="off"
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return _mlx_generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=max_new_tokens,
        temp=max(1e-5, temperature),
        verbose=False,
    ).strip()
