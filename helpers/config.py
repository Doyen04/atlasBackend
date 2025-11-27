import logging
import os
from pathlib import Path


def _split_env_list(value: str, fallback: list[str]) -> list[str]:
    candidates = [item.strip() for item in value.split(",") if item.strip()]
    return candidates or fallback

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
TMP_SPECIESNET_DIR = Path("tmp_speciesnet")
TMP_SPECIESNET_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("atlas_backend")

GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")

ALLOWED_ORIGINS = _split_env_list(os.getenv("ALLOWED_ORIGINS", "*"), ["*"])
ALLOWED_HOSTS = _split_env_list(os.getenv("ALLOWED_HOSTS", "*"), ["*"])

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

FORCE_HTTPS = os.getenv("FORCE_HTTPS", "false").lower() in {"1", "true", "yes"}
ENABLE_API_DOCS = os.getenv("ENABLE_API_DOCS", "false").lower() in {"1", "true", "yes"}
SERVICE_VERSION = os.getenv("ATLAS_BACKEND_VERSION", "1.0.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

SERVICE_NAME = "Atlas Backend"


__all__ = [
    "LOG_LEVEL",
    "logger",
    "SERVICE_NAME",
    "SERVICE_VERSION",
    "ALLOWED_TYPES",
    "ALLOWED_ORIGINS",
    "ALLOWED_HOSTS",
    "MAX_FILE_SIZE",
    "TMP_SPECIESNET_DIR",
    "GEMINI_MODEL_NAME",
    "RATE_LIMIT_REQUESTS",
    "RATE_LIMIT_WINDOW_SECONDS",
    "FORCE_HTTPS",
    "ENABLE_API_DOCS",
]
