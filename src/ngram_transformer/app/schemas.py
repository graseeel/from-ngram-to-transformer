from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ModelName = Literal["ngram", "transformer"]


def parse_model_name(value: str) -> ModelName:
    if value == "ngram":
        return "ngram"
    if value == "transformer":
        return "transformer"
    raise ValueError(f"unknown model: {value}")


class AuthRequest(BaseModel):
    email: str
    password: str = Field(min_length=6)


class GenerateRequest(BaseModel):
    model_name: ModelName = "ngram"
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


class GenerationParams(BaseModel):
    max_new_tokens: int
    temperature: float
    top_k: int | None = None
    top_p: float | None = None
    seed: int | None = None
    greedy: bool

    @classmethod
    def from_request(cls, request: GenerateRequest) -> GenerationParams:
        return cls(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=request.seed,
            greedy=request.greedy,
        )


class GenerateResponse(BaseModel):
    model_name: ModelName
    model_version_label: str
    prompt: str
    generated_text: str
    saved_generation_id: str | None = None
    generation_params: GenerationParams


class ModelInfo(BaseModel):
    name: ModelName
    ready: bool
    version_label: str | None
    notes: str
