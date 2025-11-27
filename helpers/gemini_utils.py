import os
import time
from functools import lru_cache
from io import BytesIO
from typing import Any, Optional, Dict

from PIL import Image
from google import genai
from google.genai import types, errors as genai_errors

from .config import GEMINI_MODEL_NAME, logger


GEMINI_UNAVAILABLE_MESSAGE = "Gemini model overloaded. Please try again later."


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


def call_gemini(
    prompt: str,
    pil_image: Optional[Image.Image] = None,
    response_json_schema: Optional[Dict] = None,
    *,
    max_attempts: int = 3,
    base_backoff_seconds: float = 1.5,
) -> Any:
    """Call Gemini and optionally request JSON output via a JSON Schema.

    If `response_json_schema` is provided, the request will include a config
    asking Gemini to return `application/json` that conforms to the schema.
    """
    client = get_gemini_client()

    parts = [types.Part.from_text(text=prompt)]
    if pil_image is not None:
        parts.append(_pil_image_to_part(pil_image))

    content = types.Content(role="user", parts=parts)

    last_server_error: Optional[genai_errors.ServerError] = None

    for attempt in range(1, max_attempts + 1):
        try:
            if response_json_schema is not None:
                return client.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=[content],
                    config={
                        "response_mime_type": "application/json",
                        "response_json_schema": response_json_schema,
                    },
                )

            return client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[content],
            )
        except genai_errors.ServerError as exc:  # pragma: no cover - depends on remote API
            status_code = getattr(exc, "status_code", None)
            if status_code != 503:
                raise

            last_server_error = exc
            if attempt >= max_attempts:
                break

            delay_seconds = base_backoff_seconds * attempt
            logger.warning(
                "Gemini API returned 503 (attempt %s/%s). Retrying in %.1fs.",
                attempt,
                max_attempts,
                delay_seconds,
            )
            time.sleep(delay_seconds)

    raise RuntimeError(GEMINI_UNAVAILABLE_MESSAGE) from last_server_error


def is_gemini_unavailable_error(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and str(exc) == GEMINI_UNAVAILABLE_MESSAGE


get_gemini_model = get_gemini_client


__all__ = [
    "call_gemini",
    "get_gemini_client",
    "get_gemini_model",
    "GEMINI_UNAVAILABLE_MESSAGE",
    "is_gemini_unavailable_error",
]
