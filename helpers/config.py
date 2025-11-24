import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
TMP_SPECIESNET_DIR = Path("tmp_speciesnet")
TMP_SPECIESNET_DIR.mkdir(parents=True, exist_ok=True)
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")

__all__ = [
    "ALLOWED_TYPES",
    "MAX_FILE_SIZE",
    "TMP_SPECIESNET_DIR",
    "GEMINI_MODEL_NAME",
]
