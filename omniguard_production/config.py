"""
config.py — Enterprise Production Configuration & Environment Manager.

Loads environment variables from .env if present (with fallback to OS environment),
configures Hugging Face authentication tokens, and exposes centralized configuration constants.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Set


def load_env_file(env_path: Optional[str] = None) -> None:
    """Safely loads key-value pairs from .env into os.environ without overriding explicit env vars."""
    if env_path is None:
        # Search current working directory and parent directories up to git root
        candidate = Path.cwd() / ".env"
        if not candidate.is_file():
            candidate = Path(__file__).resolve().parent.parent / ".env"
        if candidate.is_file():
            env_path = str(candidate)

    if not env_path or not os.path.isfile(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

    # Ensure HF token is mirrored across both canonical environment variable names
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if hf_token:
        os.environ.setdefault("HF_TOKEN", hf_token)
        os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", hf_token)


# Auto-load on import
load_env_file()


# Server Network Configuration
HOST: str = os.environ.get("HOST", "127.0.0.1")
PORT: int = int(os.environ.get("PORT", 8000))

# CORS & Security Settings
ALLOWED_ORIGINS_RAW: str = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:8899,http://localhost:8899"
)
ALLOWED_ORIGINS: Set[str] = {o.strip() for o in ALLOWED_ORIGINS_RAW.split(",") if o.strip()}
RATE_LIMIT_PER_MINUTE: int = int(os.environ.get("RATE_LIMIT_PER_MINUTE", 180))
MAX_CONTENT_LENGTH: int = int(os.environ.get("MAX_CONTENT_LENGTH", 1048576))

# Hugging Face Auth Token
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")

# Default Models
DEFAULT_EMBEDDING_MODEL: str = os.environ.get("DEFAULT_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
DEFAULT_CROSS_ENCODER_MODEL: str = os.environ.get("DEFAULT_CROSS_ENCODER_MODEL", "ms-marco-MiniLM-L-6-v2")
