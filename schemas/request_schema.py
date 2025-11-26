from typing import List, Optional

from fastapi import File, Form, UploadFile
from pydantic import BaseModel, Field


class GeminiAnalyzeRequest(BaseModel):
    prompt: str = Field(
        ..., description="Instructions or question to send to Google Gemini."
    )
    schema_json: Optional[str] = Field(
        default=None,
        description="Optional JSON schema string describing the desired Gemini response.",
    )
    files: List[UploadFile] = Field(
        default_factory=list,
        description="One or more images to analyze and group.",
    )

    model_config = {
        "arbitrary_types_allowed": True,
    }

    @classmethod
    def as_form(
        cls,
        prompt: str = Form(
            ..., description="Instructions or question to send to Google Gemini."
        ),
        schema_json: Optional[str] = Form(
            default=None,
            description="Optional JSON schema string describing the desired Gemini response.",
        ),
        files: List[UploadFile] = File(
            ..., description="Upload one or more images using the same form field."
        ),
    ) -> "GeminiAnalyzeRequest":
        return cls(prompt=prompt, schema_json=schema_json, files=files)
