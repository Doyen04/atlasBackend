from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Optional

from fastapi import HTTPException, UploadFile
from imghdr import what
from PIL import Image

from .config import ALLOWED_TYPES, MAX_FILE_SIZE, TMP_SPECIESNET_DIR


def is_valid_image_signature(file_bytes: bytes) -> bool:
    file_type = what(None, h=file_bytes)
    return file_type in ["jpeg", "png", "webp"]


async def read_and_validate_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPG, PNG, WEBP allowed.",
        )

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 5MB.",
        )

    if not is_valid_image_signature(content):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file signature.",
        )

    return content


def persist_temp_image(data: bytes, original_name: Optional[str]) -> Path:
    suffix = Path(original_name or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix, dir=TMP_SPECIESNET_DIR) as tmp_file:
        tmp_file.write(data)
        return Path(tmp_file.name)


def prepare_pil_image(content: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(content))
        img.load()
        return img
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Failed to decode image for Gemini.") from exc


__all__ = [
    "is_valid_image_signature",
    "read_and_validate_image",
    "persist_temp_image",
    "prepare_pil_image",
]
