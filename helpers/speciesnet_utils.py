import asyncio
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, UploadFile

from speciesnet import DEFAULT_MODEL, SpeciesNet

from .image_utils import persist_temp_image, read_and_validate_image


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


@lru_cache(maxsize=1)
def _load_speciesnet_model() -> SpeciesNet:
    logging.info("Loading SpeciesNet model: %s", DEFAULT_MODEL)
    return SpeciesNet(DEFAULT_MODEL)


def get_speciesnet_model() -> SpeciesNet:
    return _load_speciesnet_model()


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


__all__ = [
    "analyze_speciesnet_upload",
    "extract_display_name",
    "get_speciesnet_model",
    "run_speciesnet_inference",
    "summarize_prediction",
]
