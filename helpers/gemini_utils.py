import os
from functools import lru_cache
from io import BytesIO
from typing import Any

from PIL import Image
from google import genai
from google.genai import types

from .config import GEMINI_MODEL_NAME


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """Return a cached Gemini client configured with the user's API key."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is required for Gemini endpoint."
        )

    return genai.Client(api_key=api_key)


def _pil_image_to_part(pil_image: Image.Image) -> types.Part:
    buffer = BytesIO()
    save_format = (pil_image.format or "PNG").upper()
    pil_image.save(buffer, format=save_format)
    mime_type = Image.MIME.get(save_format, "image/png")
    return types.Part.from_bytes(data=buffer.getvalue(), mime_type=mime_type)


def call_gemini(prompt: str, pil_image: Image.Image) -> types.GenerateContentResponse:
    client = get_gemini_client()
    content = types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=prompt),
            _pil_image_to_part(pil_image),
        ],
    )
    return client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=[content],
    )


get_gemini_model = get_gemini_client


__all__ = ["call_gemini", "get_gemini_client", "get_gemini_model"]
