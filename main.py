import asyncio
import json
import logging
import os
from typing import Any

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from helpers import (
    GEMINI_MODEL_NAME,
    LOG_LEVEL,
    SERVICE_NAME,
    SERVICE_VERSION,
    TMP_SPECIESNET_DIR,
    analyze_speciesnet_upload,
    call_gemini,
    logger,
    prepare_pil_image,
    read_and_validate_image,
)

from schemas import GeminiAnalyzeRequest

logging.basicConfig(
    level=LOG_LEVEL,
    format="\n%(asctime)s |\n %(levelname)s |\n %(name)s |\n %(message)s",
)


app = FastAPI(
    title=SERVICE_NAME,
    version=SERVICE_VERSION,
    description=(
        "FastAPI service that validates wildlife uploads, runs SpeciesNet, and "
        "proxies Gemini for multimodal + structured responses."
    ),
)

@app.get("/", tags=["system"])
async def root() -> dict[str, Any]:
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "docs": "/docs",
        "healthz": "/healthz",
    }


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
async def create_upload_file(file: UploadFile = File(...)):
    return await analyze_speciesnet_upload(file)


@app.post("/gemini/analyze/")
async def analyze_with_gemini(
    payload: GeminiAnalyzeRequest = Depends(GeminiAnalyzeRequest.as_form),
):
    pil_image = None
    if payload.file is not None:
        content = await read_and_validate_image(payload.file)
        pil_image = prepare_pil_image(content)

    schema_dict: dict[str, Any] = {"type": "object"}
    if payload.schema_json:
        try:
            parsed_schema = json.loads(payload.schema_json)
            if not isinstance(parsed_schema, dict):
                raise ValueError("Schema must deserialize to a JSON object.")
            schema_dict = parsed_schema
        except (ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        response = await asyncio.to_thread(
            call_gemini,
            payload.prompt,
            pil_image,
            schema_dict,
        )
    except RuntimeError as config_error:
        raise HTTPException(status_code=500, detail=str(config_error)) from config_error
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gemini request failed")
        raise HTTPException(
            status_code=502,
            detail="Gemini API call failed. Check server logs for details.",
        ) from exc

    # If the generate_content path returned text (JSON string), parse it first.
    text_val = getattr(response, "text", None)
    if isinstance(text_val, str):
        try:
            response_dict = json.loads(text_val)
        except Exception:
            # Fall back to structured serialization when JSON parsing fails
            response_dict = {"text": text_val}
    elif hasattr(response, "model_dump"):
        response_dict = response.model_dump(exclude_none=True)
    else:
        to_dict_method = getattr(response, "to_dict", None)
        if callable(to_dict_method):
            response_dict = to_dict_method()
        else:
            response_dict = {"text": text_val}
    return {
        "prompt": payload.prompt,
        "schema": schema_dict,
        "gemini_model": GEMINI_MODEL_NAME,
        "response": response_dict,
    }
