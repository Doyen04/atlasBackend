import asyncio
import json
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from helpers import (
    GEMINI_MODEL_NAME,
    analyze_speciesnet_upload,
    call_gemini,
    prepare_pil_image,
    read_and_validate_image,
)


app = FastAPI()


@app.get("/")
async def run():
    return "gggg"


@app.get("/hello")
async def hello():
    return "Hello, World!"


@app.post("/analyze/")
async def create_upload_file(file: UploadFile = File(...)):
    return await analyze_speciesnet_upload(file)


@app.post("/gemini/analyze/")
async def analyze_with_gemini(
    prompt: str = Form(..., description="Instructions or question to send to Google Gemini."),
    file: UploadFile = File(...),
):
    content = await read_and_validate_image(file)
    pil_image = prepare_pil_image(content)
    try:
        # Request a JSON response from Gemini using a simple schema.
        response = await asyncio.to_thread(
            call_gemini, prompt, pil_image, {"type": "object"}
        )
    except RuntimeError as config_error:
        raise HTTPException(status_code=500, detail=str(config_error)) from config_error
    except Exception as exc:  # noqa: BLE001
        logging.exception("Gemini request failed")
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
        response_dict = response.model_dump(exclude_none=True)["candidates"]["response"]
    else:
        to_dict_method = getattr(response, "to_dict", None)
        if callable(to_dict_method):
            response_dict = to_dict_method()
        else:
            response_dict = {"text": text_val}
    return {
        "prompt": prompt,
        "gemini_model": GEMINI_MODEL_NAME,
        "response": response_dict,
    }
