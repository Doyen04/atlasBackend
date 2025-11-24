from typing import Optional
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
    file: Optional[UploadFile] = Field(
        default=None,
        description="Optional image to include with the Gemini prompt.",
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
        file: Optional[UploadFile] = File(
            default=None,
            description="Optional image to include with the Gemini prompt.",
        ),
    ) -> "GeminiAnalyzeRequest":
        return cls(prompt=prompt, schema_json=schema_json, file=file)
