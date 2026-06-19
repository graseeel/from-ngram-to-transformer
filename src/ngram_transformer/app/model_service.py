from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ngram_transformer.app.schemas import (
    GenerateRequest,
    GenerateResponse,
    GenerationParams,
    ModelInfo,
    ModelName,
)
from ngram_transformer.config import ProjectConfig, load_config
from ngram_transformer.data.tokenizer import CharacterTokenizer
from ngram_transformer.ml.checkpoints import load_transformer_checkpoint
from ngram_transformer.ml.ngram import NGramLanguageModel
from ngram_transformer.training.common import prepare_corpus


class TokenGenerator(Protocol):
    def generate(
        self,
        seed_ids: list[int],
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        greedy: bool = False,
    ) -> list[int]: ...


class ModelRunner(Protocol):
    @property
    def name(self) -> ModelName: ...

    def info(self) -> ModelInfo: ...

    def generate(self, request: GenerateRequest) -> GenerateResponse: ...


@dataclass(frozen=True)
class ReadyRunner:
    name: ModelName
    model: TokenGenerator
    tokenizer: CharacterTokenizer
    version_label: str
    notes: str

    def info(self) -> ModelInfo:
        return ModelInfo(
            name=self.name,
            ready=True,
            version_label=self.version_label,
            notes=self.notes,
        )

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        prompt_ids = self.tokenizer.encode(request.prompt)
        output_ids = self.model.generate(
            prompt_ids,
            request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=request.seed,
            greedy=request.greedy,
        )
        return GenerateResponse(
            model_name=self.name,
            model_version_label=self.version_label,
            prompt=request.prompt,
            generated_text=self.tokenizer.decode(output_ids),
            generation_params=GenerationParams.from_request(request),
        )


@dataclass(frozen=True)
class UnavailableRunner:
    name: ModelName
    notes: str

    def info(self) -> ModelInfo:
        return ModelInfo(name=self.name, ready=False, version_label=None, notes=self.notes)

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        raise ValueError(f"{self.name} is not available; train or configure a checkpoint first")


class TextGenerationService:
    def __init__(
        self,
        config: ProjectConfig,
        runners: dict[ModelName, ModelRunner],
    ) -> None:
        self.config = config
        self.runners = runners

    @classmethod
    def from_config(cls, config_path: str | Path) -> TextGenerationService:
        config = load_config(config_path)
        prepared = prepare_corpus(config.data)
        ngram_artifact = Path(os.getenv("NGRAM_MODEL_PATH", "artifacts/ngram/ngram_model.json"))
        if ngram_artifact.exists():
            ngram_model = NGramLanguageModel.load(ngram_artifact)
            ngram_version = f"ngram:{ngram_artifact.as_posix()}"
        else:
            ngram_model = NGramLanguageModel(config.ngram, prepared.tokenizer.vocab_size)
            ngram_model.fit(prepared.train_ids)
            ngram_version = f"ngram:n={config.ngram.n}:add_k={config.ngram.add_k}:in-memory"

        runners: dict[ModelName, ModelRunner] = {}
        runners["ngram"] = ReadyRunner(
            name="ngram",
            model=ngram_model,
            tokenizer=prepared.tokenizer,
            version_label=ngram_version,
            notes="Add-k smoothed character N-gram model.",
        )
        checkpoint_path = Path(
            os.getenv(
                "TRANSFORMER_CHECKPOINT_PATH",
                "artifacts/transformer/transformer_checkpoint.pt",
            ),
        )
        if checkpoint_path.exists():
            model, tokenizer, metadata = load_transformer_checkpoint(checkpoint_path)
            runners["transformer"] = ReadyRunner(
                name="transformer",
                model=model,
                tokenizer=tokenizer,
                version_label=f"transformer:{checkpoint_path.as_posix()}:step={metadata.step}",
                notes="Small decoder-only Transformer checkpoint.",
            )
        else:
            runners["transformer"] = UnavailableRunner(
                name="transformer",
                notes="Load artifacts/transformer/transformer_checkpoint.pt after training.",
            )
        return cls(config, runners)

    def list_models(self) -> list[ModelInfo]:
        return [runner.info() for runner in self.runners.values()]

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        return self.runners[request.model_name].generate(request)
