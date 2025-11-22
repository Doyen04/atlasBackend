import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from imghdr import what

from speciesnet import SpeciesNet, DEFAULT_MODEL

app = FastAPI()


def is_valid_image_signature(file_bytes: bytes) -> bool:
    file_type = what(None, h=file_bytes)
    return file_type in ["jpeg", "png", "webp"]

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
TMP_SPECIESNET_DIR = Path("tmp_speciesnet")
TMP_SPECIESNET_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_speciesnet_model() -> SpeciesNet:
    """Load SpeciesNet once and reuse it across requests."""
    logging.info("Loading SpeciesNet model: %s", DEFAULT_MODEL)
    return SpeciesNet(DEFAULT_MODEL)


def _persist_temp_image(data: bytes, original_name: str | None) -> Path:
    """Write upload bytes to disk so SpeciesNet can read them from a filepath."""
    suffix = Path(original_name or "upload.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        suffix = ".jpg"
    with NamedTemporaryFile(delete=False, suffix=suffix, dir=TMP_SPECIESNET_DIR) as tmp_file:
        tmp_file.write(data)
        return Path(tmp_file.name)


def _predict_sync(image_path: Path) -> Dict[str, Any] | None:
    """Run SpeciesNet prediction in a synchronous context."""
    instances = {"instances": [{"filepath": str(image_path)}]}
    return get_speciesnet_model().predict(
        instances_dict=instances,
        run_mode="single_thread",
        batch_size=1,
        progress_bars=False,
    )


async def run_speciesnet_inference(image_path: Path) -> Dict[str, Any] | None:
    return await asyncio.to_thread(_predict_sync, image_path)


def _summarize_prediction(predictions_dict: Dict[str, Any]|None) -> Dict[str, Any]:
    predictions: List[Dict[str, Any]] = predictions_dict.get("predictions", [])  # type: ignore[assignment]
    if not predictions:
        raise RuntimeError("SpeciesNet returned no predictions.")
    first = predictions[0]
    classifications = first.get("classifications") or {}
    top_classes = [
        {"label": label, "score": score}
        for label, score in zip(classifications.get("classes", []), classifications.get("scores", []))
    ]
    return {
        "prediction": first.get("prediction"),
        "prediction_score": first.get("prediction_score"),
        "prediction_source": first.get("prediction_source"),
        "model_version": first.get("model_version"),
        "top_classes": top_classes,
        "detections": first.get("detections", []),
        "failures": first.get("failures"),
    }

@app.get("/")
async def run():
    return 'gggg'

@app.get("/hello")
async def hello():
    return "Hello, World!"


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        print(f"Invalid file type: {file.content_type}")
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPG, PNG, WEBP allowed."
        )
    
    # Read file
    content = await file.read()

    # Validate size
    if len(content) > MAX_FILE_SIZE:
        print(f"File too large: {len(content)} bytes")
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 5MB."
        )

    # OPTIONAL: Validate file signature (magic number)
    if not is_valid_image_signature(content):
        print("Invalid image file signature.")
        raise HTTPException(
            status_code=400,
            detail="Invalid image file signature."
        )

    temp_image_path = _persist_temp_image(content, file.filename)
    try:
        try:
            predictions_dict = await run_speciesnet_inference(temp_image_path)
            speciesnet_summary = _summarize_prediction(predictions_dict)
        except Exception as exc:  # noqa: BLE001 - bubble up as HTTP error
            logging.exception("SpeciesNet inference failed")
            raise HTTPException(
                status_code=500,
                detail="SpeciesNet inference failed. Check server logs for details."
            ) from exc
    finally:
        temp_image_path.unlink(missing_ok=True)

    return {
        "filename": file.filename,
        "content_size": len(content),
        "speciesnet": speciesnet_summary,
        "predictions": predictions_dict
    }
