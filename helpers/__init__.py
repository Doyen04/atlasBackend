import asyncio
import logging
import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, UploadFile
from imghdr import what
import google.generativeai as genai
from PIL import Image
from speciesnet import DEFAULT_MODEL, SpeciesNet

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
TMP_SPECIESNET_DIR = Path("tmp_speciesnet")
TMP_SPECIESNET_DIR.mkdir(parents=True, exist_ok=True)
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")


def is_valid_image_signature(file_bytes: bytes) -> bool:
    file_type = what(None, h=file_bytes)
    return file_type in ["jpeg", "png", "webp"]


@lru_cache(maxsize=1)
def get_speciesnet_model() -> SpeciesNet:
    logging.info("Loading SpeciesNet model: %s", DEFAULT_MODEL)
    return SpeciesNet(DEFAULT_MODEL)


@lru_cache(maxsize=1)
def get_gemini_model() -> Any:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is required for Gemini endpoint."
        )
    genai.configure(api_key=api_key)  # type: ignore[attr-defined]
    return genai.GenerativeModel(GEMINI_MODEL_NAME)  # type: ignore[attr-defined]


async def read_and_validate_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        print(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPG, PNG, WEBP allowed.",
        )

    content = await file.read()

    if len(content) > MAX_FILE_SIZE:
        print(f"File too large: {len(content)} bytes")
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 5MB.",
        )

    if not is_valid_image_signature(content):
        print("Invalid image file signature.")
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


def _predict_sync(image_path: Path) -> Dict[str, Any] | None:
    instances = {"instances": [{"filepath": str(image_path)}]}
    return get_speciesnet_model().predict(
        instances_dict=instances,
        run_mode="single_thread",
        batch_size=1,
        progress_bars=False,
    )


async def run_speciesnet_inference(image_path: Path) -> Dict[str, Any] | None:
    return await asyncio.to_thread(_predict_sync, image_path)


def extract_display_name(label: Optional[str]) -> Optional[str]:
    if not label:
        return None
    parts = [segment.strip() for segment in label.split(";") if segment.strip()]
    return parts[-1] if parts else label


def summarize_prediction(predictions_dict: Dict[str, Any] | None) -> Dict[str, Any]:
    predictions: List[Dict[str, Any]] = predictions_dict.get("predictions", [])  # type: ignore[assignment]
    if not predictions:
        raise RuntimeError("SpeciesNet returned no predictions.")
    first = predictions[0]
    classifications = first.get("classifications") or {}
    top_classes = []
    for label, score in zip(
        classifications.get("classes", []),
        classifications.get("scores", []),
    ):
        top_classes.append(
            {
                "label_raw": label,
                "display_name": extract_display_name(label),
                "score": score,
            }
        )

    detections = first.get("detections", [])
    for detection in detections:
        detection["label_display"] = extract_display_name(detection.get("label"))

    prediction_label = first.get("prediction")
    return {
        "prediction": prediction_label,
        "prediction_display_name": extract_display_name(prediction_label),
        "prediction_score": first.get("prediction_score"),
        "prediction_source": first.get("prediction_source"),
        "model_version": first.get("model_version"),
        "top_classes": top_classes,
        "detections": detections,
        "best_class": top_classes[0] if top_classes else None,
        "failures": first.get("failures"),
    }


async def analyze_speciesnet_upload(file: UploadFile) -> Dict[str, Any]:
    content = await read_and_validate_image(file)
    temp_image_path = persist_temp_image(content, file.filename)
    try:
        try:
            predictions_dict = await run_speciesnet_inference(temp_image_path)
            speciesnet_summary = summarize_prediction(predictions_dict)
        except Exception as exc:  # noqa: BLE001
            logging.exception("SpeciesNet inference failed")
            raise HTTPException(
                status_code=500,
                detail="SpeciesNet inference failed. Check server logs for details.",
            ) from exc
    finally:
        temp_image_path.unlink(missing_ok=True)

    return {
        "filename": file.filename,
        "content_size": len(content),
        "speciesnet": speciesnet_summary,
    }


def prepare_pil_image(content: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(content))
        img.load()
        return img
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Failed to decode image for Gemini.") from exc


def call_gemini(prompt: str, pil_image: Image.Image):
    model = get_gemini_model()
    return model.generate_content([prompt, pil_image])


__all__ = [
    "ALLOWED_TYPES",
    "MAX_FILE_SIZE",
    "TMP_SPECIESNET_DIR",
    "GEMINI_MODEL_NAME",
    "analyze_speciesnet_upload",
    "call_gemini",
    "extract_display_name",
    "get_gemini_model",
    "get_speciesnet_model",
    "is_valid_image_signature",
    "prepare_pil_image",
    "read_and_validate_image",
    "run_speciesnet_inference",
    "summarize_prediction",
]
