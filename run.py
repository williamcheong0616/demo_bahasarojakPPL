#!/usr/bin/env python3
"""
Driver script for Bahasa Rojak AI Demo.

Detects the environment, sets the right backend, optionally ensures Ollama is
running, then starts the uvicorn server and opens the browser.

Usage:
  python run.py                  # auto-detect backend
  python run.py --backend ollama # force Ollama SLM
  python run.py --backend mlx    # force MLX SLM
  python run.py --backend cuda   # force CUDA SLM
  python run.py --port 8080      # custom port
  python run.py --no-browser     # don't open browser automatically
  python run.py --https          # enable HTTPS with a self-signed cert (required for mic over VPN/LAN)
  python run.py --https --cert cert.pem --key key.pem  # bring your own cert
"""

import argparse
import platform
import shutil
import subprocess
import sys
import time
import webbrowser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print(msg: str, tag: str = "run"):
    print(f"[{tag}] {msg}", flush=True)


def _has_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def _detect_backend() -> str:
    """Pick the best SLM backend for the current machine."""
    # Check MLX (Apple Silicon)
    try:
        import mlx.core  # noqa: F401
        return "mlx"
    except ImportError:
        pass

    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    # Fallback — let Ollama handle it if available
    if _has_cmd("ollama"):
        return "ollama"

    return "cuda"   # will warn at model load, but keep going


def _check_ffmpeg():
    if not _has_cmd("ffmpeg"):
        _print("WARNING: ffmpeg not found. ASR will fail without it.", "run")
        _print("  macOS:  brew install ffmpeg", "run")
        _print("  Linux:  sudo apt install ffmpeg", "run")


def _ensure_cert(cert_path: str, key_path: str):
    """Generate a self-signed cert/key pair if either file is missing."""
    import os
    if os.path.isfile(cert_path) and os.path.isfile(key_path):
        _print(f"Using existing cert: {cert_path}", "run")
        return

    if not _has_cmd("openssl"):
        _print("ERROR: 'openssl' not found — cannot generate a self-signed cert.", "run")
        _print("Install openssl, or provide --cert / --key pointing to existing files.", "run")
        sys.exit(1)

    _print("Generating self-signed certificate (valid 365 days)...", "run")
    result = subprocess.run([
        "openssl", "req", "-x509",
        "-newkey", "rsa:4096",
        "-keyout", key_path,
        "-out", cert_path,
        "-days", "365",
        "-nodes",
        "-subj", "/CN=rojak-demo",
    ], capture_output=True)

    if result.returncode != 0:
        _print("ERROR: openssl failed:", "run")
        _print(result.stderr.decode(), "run")
        sys.exit(1)

    _print(f"Certificate written to {cert_path}", "run")
    _print(f"Private key  written to {key_path}", "run")


def _ensure_ollama(model: str):
    """Make sure Ollama is running and the model is pulled."""
    if not _has_cmd("ollama"):
        _print("ERROR: 'ollama' command not found.", "run")
        _print("Install from https://ollama.com then re-run.", "run")
        sys.exit(1)

    # Check if the server is already up
    import httpx
    server_up = False
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        server_up = r.status_code == 200
    except Exception:
        pass

    if not server_up:
        _print("Ollama server not running — starting it in the background...", "run")
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait up to 8 s for the server to come up
        for _ in range(8):
            time.sleep(1)
            try:
                r = httpx.get("http://localhost:11434/api/tags", timeout=2)
                if r.status_code == 200:
                    _print("Ollama server ready.", "run")
                    break
            except Exception:
                pass
        else:
            _print("ERROR: Ollama server did not start in time.", "run")
            sys.exit(1)

    # Check if the model is already pulled
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5)
        available = [m["name"] for m in r.json().get("models", [])]
        base = model.split(":")[0]
        matched = [m for m in available if m.split(":")[0] == base or m == model]
    except Exception:
        matched = []

    if not matched:
        _print(f"Model '{model}' not found locally — pulling now...", "run")
        _print("(This is a one-time download, may take a few minutes.)", "run")
        result = subprocess.run(["ollama", "pull", model])
        if result.returncode != 0:
            _print(f"ERROR: Failed to pull '{model}'.", "run")
            sys.exit(1)
        _print(f"Model '{model}' ready.", "run")
    else:
        _print(f"Model ready: {matched[0]}", "run")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the Bahasa Rojak AI Demo")
    parser.add_argument(
        "--backend", choices=["auto", "ollama", "mlx", "cuda"], default="auto",
        help="SLM backend (default: auto-detect)"
    )
    parser.add_argument(
        "--ollama-model", default="gemma3:4b",
        help="Ollama model tag to use (default: gemma3:4b)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn auto-reload (dev)")
    parser.add_argument("--https", action="store_true",
                        help="Serve over HTTPS with a self-signed cert (required for mic access over VPN/LAN)")
    parser.add_argument("--cert", default="cert.pem", help="Path to TLS certificate (default: cert.pem)")
    parser.add_argument("--key",  default="key.pem",  help="Path to TLS private key  (default: key.pem)")
    args = parser.parse_args()

    # ── Detect backend ───────────────────────────────────────────────────────
    backend = args.backend if args.backend != "auto" else _detect_backend()
    _print(f"SLM backend: {backend}", "run")

    # ── System checks ────────────────────────────────────────────────────────
    _check_ffmpeg()

    if backend == "ollama":
        _ensure_ollama(args.ollama_model)

    # ── Build environment for the server process ─────────────────────────────
    import os
    env = os.environ.copy()
    env["SLM_BACKEND"]  = backend
    env["OLLAMA_MODEL"] = args.ollama_model

    # ── TLS / HTTPS setup ────────────────────────────────────────────────────
    use_https = args.https
    if use_https:
        _ensure_cert(args.cert, args.key)

    scheme = "https" if use_https else "http"

    # ── Start uvicorn ────────────────────────────────────────────────────────
    url = f"{scheme}://{args.host}:{args.port}"
    _print(f"Starting server at {url}", "run")
    if use_https:
        _print("NOTE: Your browser will show a security warning for the self-signed cert.", "run")
        _print("      Click 'Advanced' → 'Proceed' (or 'Accept the Risk') to continue.", "run")
    _print("Startup takes 2–5 min while models load. Watch for '[startup] ... ready.' lines.", "run")

    cmd = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if use_https:
        cmd += ["--ssl-certfile", args.cert, "--ssl-keyfile", args.key]
    if args.reload:
        cmd.append("--reload")

    # Open browser — skip on headless Linux servers (no display = no GUI browser)
    has_display = (
        platform.system() != "Linux"
        or bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    )
    if not args.no_browser and has_display:
        import threading
        def _open_browser():
            time.sleep(3)
            webbrowser.open(url)
        threading.Thread(target=_open_browser, daemon=True).start()
    elif not has_display:
        _print(f"Headless server detected — open {url} in your local browser", "run")
        if use_https:
            _print("Tip: Access directly via https://<server-ip>:{port} over VPN".format(port=args.port), "run")
        else:
            _print("Tip: SSH tunnel with:  ssh -L {port}:localhost:{port} <user>@<server>".format(port=args.port), "run")

    try:
        subprocess.run(cmd, env=env)
    except KeyboardInterrupt:
        _print("Shutting down.", "run")


if __name__ == "__main__":
    main()
