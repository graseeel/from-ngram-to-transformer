from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class AuthRequest(BaseModel):
    email: str
    password: str = Field(min_length=6)


class GenerateRequest(BaseModel):
    model_name: Literal["ngram", "transformer"] = "ngram"
    prompt: str = Field(min_length=1, max_length=2_000)
    max_new_tokens: int = Field(default=120, gt=0, le=1_000)
    temperature: float = Field(default=1.0, gt=0, le=5.0)
    top_k: int | None = Field(default=None, gt=0, le=500)
    top_p: float | None = Field(default=None, gt=0, le=1.0)
    seed: int | None = None
    greedy: bool = False
    save: bool = True

    @field_validator("prompt")
    @classmethod
    def prompt_must_have_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("prompt must contain non-whitespace text")
        return stripped


class GenerateResponse(BaseModel):
    model_name: str
    model_version_label: str
    prompt: str
    generated_text: str
    saved_generation_id: str | None = None
    generation_params: dict[str, Any]


class ModelInfo(BaseModel):
    name: str
    ready: bool
    version_label: str | None
    notes: str
