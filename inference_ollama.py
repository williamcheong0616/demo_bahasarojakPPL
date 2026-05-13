"""
Ollama SLM backend — calls a locally running Ollama server via HTTP.

Matches the load_model / generate interface of inference.py and inference_mlx.py
so main.py can swap backends transparently.

Zero model RAM in the app process — Ollama manages its own memory.

Setup:
  1. Install Ollama: https://ollama.com
  2. Pull the model: ollama pull gemma3:2b
  3. Set SLM_BACKEND=ollama (or let main.py auto-detect)
"""

import httpx

OLLAMA_URL = "http://localhost:11434"
_HTTP_TIMEOUT = 120   # seconds — generation can be slow on first run


def load_model(adapter_path: str, model_name: str):
    """
    Verify Ollama is reachable and the requested model is available.
    Returns (model_name, None) — 'tokenizer' slot is unused for Ollama.
    """
    try:
        r = httpx.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except Exception as e:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. "
            f"Is it running? (ollama serve)  Error: {e}"
        )

    available = [m["name"] for m in r.json().get("models", [])]
    # Ollama tags can be "gemma3:2b" or "gemma3:2b-instruct-q4_K_M" etc.
    # Match on the base name before any colon or extra suffix.
    base = model_name.split(":")[0]
    matched = [m for m in available if m.split(":")[0] == base or m == model_name]
    if not matched:
        raise RuntimeError(
            f"Model '{model_name}' not found in Ollama. "
            f"Run: ollama pull {model_name}\n"
            f"Available models: {available or '(none)'}"
        )

    print(f"[Ollama] Using model: {matched[0]}")
    return model_name, None   # (model_name_str, tokenizer=None)


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


def generate(model_name: str, _tokenizer, instruction: str, input_ctx: str = "",
             max_new_tokens: int = 512, temperature: float = 0.7) -> str:
    content = instruction + (f"\n\n{input_ctx}" if input_ctx else "")
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        "stream": False,
        "options": {
            "num_predict": max_new_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    }
    r = httpx.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=_HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()
