import asyncio
import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from helpers import (
    ENABLE_API_DOCS,
    GEMINI_MODEL_NAME,
    GEMINI_UNAVAILABLE_MESSAGE,
    LOG_LEVEL,
    SERVICE_NAME,
    SERVICE_VERSION,
    TMP_SPECIESNET_DIR,
    analyze_speciesnet_upload,
    call_gemini,
    configure_rate_limiting,
    is_gemini_unavailable_error,
    logger,
    prepare_pil_image,
    read_and_validate_image,
)
from middlewares import configure_middlewares

from schemas import GeminiAnalyzeRequest

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

DOCS_URL = "/docs" if ENABLE_API_DOCS else None
REDOC_URL = "/redoc" if ENABLE_API_DOCS else None
OPENAPI_URL = "/openapi.json" if ENABLE_API_DOCS else None

DEFAULT_GROUPING_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "category_label": {
            "type": "string",
            "description": "High-level label shared by related images (species, scene, etc.)",
        },
        "analysis": {
            "type": "string",
            "description": "Detailed description tailored to the prompt",
        },
        "details": {
            "type": "object",
            "description": "Optional structured fields returned by Gemini.",
        },
    },
    "required": ["category_label", "analysis"],
}

GROUP_LABEL_FIELDS = (
    "category_label",
    "group_label",
    "label",
    "category",
    "cluster",
    "species",
    "name",
    "title",
)


def _extract_label_from_mapping(candidate: Dict[str, Any]) -> Optional[str]:
    for field in GROUP_LABEL_FIELDS:
        value = candidate.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _serialize_gemini_response(response: Any) -> Dict[str, Any]:
    text_val = getattr(response, "text", None)
    if isinstance(text_val, str):
        try:
            return json.loads(text_val)
        except Exception:
            return {"text": text_val}

    if hasattr(response, "model_dump"):
        return response.model_dump(exclude_none=True)

    # to_dict_method = getattr(response, "to_dict", None)
    # if callable(to_dict_method):
    #     return to_dict_method()

    return {"text": text_val}


def _derive_group_label(response_dict: Dict[str, Any], fallback: str) -> str:
    direct_label = _extract_label_from_mapping(response_dict)
    if direct_label:
        return direct_label

    nested_candidates: List[Any] = []
    results = response_dict.get("results")
    if isinstance(results, list):
        nested_candidates.extend(results)

    details = response_dict.get("details")
    if isinstance(details, dict):
        nested_candidates.append(details)

    for candidate in nested_candidates:
        if isinstance(candidate, dict):
            nested_label = _extract_label_from_mapping(candidate)
            if nested_label:
                return nested_label

    return fallback

app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description=(
        "FastAPI service that validates wildlife uploads, runs SpeciesNet, and "
        "proxies Gemini for multimodal + structured responses."
    ),
    docs_url=DOCS_URL,
    redoc_url=REDOC_URL,
    openapi_url=OPENAPI_URL,
)

configure_middlewares(app)
configure_rate_limiting(app)

@app.get("/", tags=["system"])
async def root() -> dict[str, Any]:
    payload = {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "healthz": "/healthz",
    }
    if DOCS_URL:
        payload["docs"] = DOCS_URL
    return payload


@app.get("/healthz", tags=["system"])
async def health_check() -> dict[str, Any]:
    api_key_configured = bool(os.getenv("GOOGLE_API_KEY"))
    tmp_dir_ready = TMP_SPECIESNET_DIR.exists()
    status = "ok" if api_key_configured and tmp_dir_ready else "degraded"
    return {
        "status": status,
        "gemini_api_configured": api_key_configured,
        "temp_dir_ready": tmp_dir_ready,
    }

@app.post("/analyze/")
async def create_upload_file(
    file: UploadFile = File(...),
):
    return await analyze_speciesnet_upload(file)


@app.post("/gemini/analyze/")
async def analyze_with_gemini(
    payload: GeminiAnalyzeRequest = Depends(GeminiAnalyzeRequest.as_form),
):
    if not payload.files:
        raise HTTPException(status_code=400, detail="At least one image file is required.")

    schema_dict: Dict[str, Any] = deepcopy(DEFAULT_GROUPING_SCHEMA)
    if payload.schema_json:
        try:
            parsed_schema = json.loads(payload.schema_json)
            if not isinstance(parsed_schema, dict):
                raise ValueError("Schema must deserialize to a JSON object.")
            schema_dict = parsed_schema
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    grouped_results: Dict[str, Dict[str, Any]] = {}

    for index, upload in enumerate(payload.files):
        content = await read_and_validate_image(upload)
        pil_image = prepare_pil_image(content)
        contextual_prompt = (
            f"{payload.prompt}\nImage index: {index + 1}\nFilename: {upload.filename or 'upload'}"
        )
        try:
            response = await asyncio.to_thread(
                call_gemini,
                contextual_prompt,
                pil_image,
                schema_dict,
            )
        except RuntimeError as runtime_error:
            if is_gemini_unavailable_error(runtime_error):
                logger.warning("Gemini overloaded for %s: %s", upload.filename, runtime_error)
                raise HTTPException(
                    status_code=503,
                    detail=GEMINI_UNAVAILABLE_MESSAGE,
                    headers={"Retry-After": "30"},
                ) from runtime_error

            raise HTTPException(status_code=500, detail=str(runtime_error)) from runtime_error
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gemini request failed for %s", upload.filename)
            raise HTTPException(
                status_code=502,
                detail="Gemini API call failed. Check server logs for details.",
            ) from exc

        response_dict = _serialize_gemini_response(response)
        group_key = _derive_group_label(response_dict, f"group_{index + 1}")

        group_entry = grouped_results.setdefault(
            group_key,
            {
                "summary": None,
                "items": [],
            },
        )

        if group_entry["summary"] is None:
            group_entry["summary"] = response_dict

        group_entry["items"].append(
            {
                "index": index,
                "filename": upload.filename,
                "content_size": len(content),
            }
        )

    groups_payload = []
    for key, data in grouped_results.items():
        items = data["items"]
        groups_payload.append(
            {
                "group": key,
                "count": len(items),
                "summary": data["summary"],
                "items": items,
            }
        )

    return {
        "prompt": payload.prompt,
        "schema": schema_dict,
        "gemini_model": GEMINI_MODEL_NAME,
        "total_images": len(payload.files),
        "groups": groups_payload,
    }
