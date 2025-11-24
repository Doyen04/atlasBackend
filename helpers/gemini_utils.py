import os
from typing import Any

import google.generativeai as genai

from .config import GEMINI_MODEL_NAME


def get_gemini_model() -> Any:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is required for Gemini endpoint."
        )
    genai.configure(api_key=api_key)  # type: ignore[attr-defined]
    return genai.GenerativeModel(GEMINI_MODEL_NAME)  # type: ignore[attr-defined]


def call_gemini(prompt: str, pil_image) -> Any:
    model = get_gemini_model()
    return model.generate_content([prompt, pil_image])


__all__ = ["call_gemini", "get_gemini_model"]
