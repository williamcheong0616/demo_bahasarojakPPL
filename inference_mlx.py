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
