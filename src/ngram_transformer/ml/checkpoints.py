from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ngram_transformer.config import TransformerConfig
from ngram_transformer.data.tokenizer import CharacterTokenizer
from ngram_transformer.ml.transformer import TransformerLanguageModel


def save_transformer_checkpoint(
    path: str | Path,
    model: TransformerLanguageModel,
    tokenizer: CharacterTokenizer,
    *,
    step: int,
    metrics: dict[str, float],
    optimizer_state: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": asdict(model.config),
        "tokenizer": {
            "unknown_token": tokenizer.unknown_token,
            "token_to_id": tokenizer.token_to_id,
        },
        "step": step,
        "metrics": metrics,
        "optimizer_state": optimizer_state,
    }
    torch.save(payload, target)


def load_transformer_checkpoint(
    path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[TransformerLanguageModel, CharacterTokenizer, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    config = TransformerConfig(**payload["config"])
    model = TransformerLanguageModel(config)
    model.load_state_dict(payload["model_state"])
    tokenizer_payload = payload["tokenizer"]
    tokenizer = CharacterTokenizer(
        token_to_id={
            str(token): int(index)
            for token, index in tokenizer_payload["token_to_id"].items()
        },
        unknown_token=str(tokenizer_payload["unknown_token"]),
    )
    metadata = {
        "step": int(payload["step"]),
        "metrics": dict(payload["metrics"]),
        "optimizer_state": payload.get("optimizer_state"),
    }
    return model, tokenizer, metadata
