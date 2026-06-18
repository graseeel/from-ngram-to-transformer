from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ngram_transformer.app.schemas import GenerateRequest, GenerateResponse, ModelInfo
from ngram_transformer.config import ProjectConfig, load_config
from ngram_transformer.data.tokenizer import CharacterTokenizer
from ngram_transformer.ml.checkpoints import load_transformer_checkpoint
from ngram_transformer.ml.ngram import NGramLanguageModel
from ngram_transformer.ml.transformer import TransformerLanguageModel
from ngram_transformer.training.common import prepare_corpus


@dataclass
class LoadedTransformer:
    model: TransformerLanguageModel
    tokenizer: CharacterTokenizer
    version_label: str


class TextGenerationService:
    def __init__(
        self,
        config: ProjectConfig,
        tokenizer: CharacterTokenizer,
        ngram_model: NGramLanguageModel,
        ngram_version_label: str,
        transformer: LoadedTransformer | None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.ngram_model = ngram_model
        self.ngram_version_label = ngram_version_label
        self.transformer = transformer

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

        transformer = None
        checkpoint_path = Path(
            os.getenv(
                "TRANSFORMER_CHECKPOINT_PATH",
                "artifacts/transformer/transformer_checkpoint.pt",
            ),
        )
        if checkpoint_path.exists():
            model, tokenizer, metadata = load_transformer_checkpoint(checkpoint_path)
            transformer = LoadedTransformer(
                model=model,
                tokenizer=tokenizer,
                version_label=f"transformer:{checkpoint_path.as_posix()}:step={metadata['step']}",
            )
        return cls(config, prepared.tokenizer, ngram_model, ngram_version, transformer)

    def list_models(self) -> list[ModelInfo]:
        transformer = self.transformer
        return [
            ModelInfo(
                name="ngram",
                ready=True,
                version_label=self.ngram_version_label,
                notes="Add-k smoothed character N-gram model.",
            ),
            ModelInfo(
                name="transformer",
                ready=transformer is not None,
                version_label=transformer.version_label if transformer is not None else None,
                notes="Load artifacts/transformer/transformer_checkpoint.pt after training.",
            ),
        ]

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        params = {
            "max_new_tokens": request.max_new_tokens,
            "temperature": request.temperature,
            "top_k": request.top_k,
            "top_p": request.top_p,
            "seed": request.seed,
            "greedy": request.greedy,
        }
        if request.model_name == "ngram":
            prompt_ids = self.tokenizer.encode(request.prompt)
            output_ids = self.ngram_model.generate(
                prompt_ids,
                request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                seed=request.seed,
                greedy=request.greedy,
            )
            return GenerateResponse(
                model_name="ngram",
                model_version_label=self.ngram_version_label,
                prompt=request.prompt,
                generated_text=self.tokenizer.decode(output_ids),
                generation_params=params,
            )
        if self.transformer is None:
            raise ValueError(
                "transformer checkpoint is not available; train or configure one first",
            )
        prompt_ids = self.transformer.tokenizer.encode(request.prompt)
        output_ids = self.transformer.model.generate(
            prompt_ids,
            request.max_new_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            seed=request.seed,
            greedy=request.greedy,
        )
        return GenerateResponse(
            model_name="transformer",
            model_version_label=self.transformer.version_label,
            prompt=request.prompt,
            generated_text=self.transformer.tokenizer.decode(output_ids),
            generation_params=params,
        )
