---
title: "SpeciesNet FastAPI Space"
emoji: 🐾
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
app_file: app.py
pinned: false
---

# Atlas Backend

FastAPI microservice that validates uploaded wildlife images and extracts animal
information with [SpeciesNet](https://pypi.org/project/speciesnet/).

## Quick start

1. Create / activate a Python 3.12+ environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Launch the API: `uvicorn main:app --reload`.

## `/uploadfile/` endpoint

- Accepts a multipart form upload with the `file` field.
- Rejects non-image media types, files over 5 MB, or images with invalid magic
  numbers.
- Writes the image to `tmp_speciesnet/`, invokes `SpeciesNet(DEFAULT_MODEL)` on
  the image path, then removes the temporary file.
- Responds with the filename, byte size, best SpeciesNet prediction, top-5
  classes, and any detections returned by the ensemble (animals, humans,
  vehicles, etc.).

Example cURL invocation:

```bash
curl -X POST http://localhost:8000/uploadfile/ \
  -F "file=@/path/to/camera-trap.jpg"
```

Example JSON fragment:

```json
{
    "filename": "camera-trap.jpg",
    "content_size": 123456,
    "speciesnet": {
        "prediction": "odocoileus_virginianus",
        "prediction_display_name": "odocoileus_virginianus",
        "prediction_score": 0.97,
        "top_classes": [
            {"label_raw": "odocoileus_virginianus", "display_name": "odocoileus_virginianus", "score": 0.97},
            {"label_raw": "odocoileus", "display_name": "odocoileus", "score": 0.99}
        ],
        "detections": [
            {"label": "animal", "conf": 0.92, "bbox": [0.12, 0.33, 0.56, 0.41]}
        ]
    }
}
```

## `/analyze/` endpoint (prompt + image)

- Accepts a multipart form upload with two fields:
  - `prompt` (`text/plain` form field) describing the question/instructions.
  - `file` (image) identical requirements as `/uploadfile/`.
- Returns the exact SpeciesNet analysis plus the original prompt, so you can tie
  downstream workflows or LLM calls to the same metadata.

Example request:

```bash
curl -X POST http://localhost:8000/analyze/ \
  -F "prompt=Summarize this animal" \
  -F "file=@/path/to/camera-trap.jpg"
```

## Operational notes

- The first request downloads the SpeciesNet weights (default model `kaggle:google/speciesnet/pyTorch/v4.0.1a`), so expect a slow cold start.
- Inference runs inside a worker thread to keep the FastAPI event loop
  responsive.
- Temporary files live under `tmp_speciesnet/` and are deleted immediately after
  each inference.
- Refer to the [SpeciesNet PyPI documentation](https://pypi.org/project/speciesnet/)
  for detailed information about inputs, outputs, geofencing, and the detection/
  classification ensemble used by this service.

  ### `/gemini/analyze/` endpoint

  - Requires a Google AI Studio API key exposed as `GOOGLE_API_KEY` (set
    optionally `GEMINI_MODEL`, default `gemini-1.5-flash`).
  - Accepts `prompt` + `file` form fields and proxies both to Google Gemini via the
    `google-genai` SDK.
  - Returns JSON containing the submitted prompt, the model used, and the raw
    Gemini response (serialized dictionary).

  Example:

  ```bash
  export GOOGLE_API_KEY="sk-..."
  curl -X POST http://localhost:8000/gemini/analyze/ \
    -F "prompt=Describe this animal" \
    -F "file=@/path/to/camera-trap.jpg"
  ```

## Deploy with Docker

```bash
docker build -t atlas-backend .
docker run --rm -p 8000:8000 atlas-backend
```

If you are deploying to Hugging Face Spaces (Docker SDK), this repository's
`DockerFile` already exposes port `8000`, matching the `app_port` metadata and
the FastAPI configuration. Adjust environment variables or Uvicorn arguments as
needed for your hosting provider.
