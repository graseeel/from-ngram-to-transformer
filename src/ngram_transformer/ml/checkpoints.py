from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from ngram_transformer.config import TransformerConfig
from ngram_transformer.data.tokenizer import CharacterTokenizer
from ngram_transformer.ml.transformer import TransformerLanguageModel


@dataclass(frozen=True)
class CheckpointMetadata:
    step: int
    metrics: dict[str, float]
    optimizer_state: object | None


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint field {name!r} must be a mapping")
    return value


def _int_value(value: object, name: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"checkpoint field {name!r} must be an integer")


def _optional_int_value(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _int_value(value, name)


def _float_value(value: object, name: str) -> float:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"checkpoint field {name!r} must be numeric")


def _config_from_payload(value: object) -> TransformerConfig:
    payload = _mapping(value, "config")
    return TransformerConfig(
        block_size=_int_value(payload.get("block_size"), "config.block_size"),
        n_layer=_int_value(payload.get("n_layer"), "config.n_layer"),
        n_head=_int_value(payload.get("n_head"), "config.n_head"),
        n_embd=_int_value(payload.get("n_embd"), "config.n_embd"),
        dropout=_float_value(payload.get("dropout"), "config.dropout"),
        vocab_size=_optional_int_value(payload.get("vocab_size"), "config.vocab_size"),
    )


def _float_metrics(value: object) -> dict[str, float]:
    metrics = _mapping(value, "metrics")
    return {str(key): _float_value(metric, f"metrics.{key}") for key, metric in metrics.items()}


def _tokenizer_from_payload(value: object) -> CharacterTokenizer:
    payload = _mapping(value, "tokenizer")
    token_to_id = _mapping(payload.get("token_to_id"), "tokenizer.token_to_id")
    unknown_token = payload.get("unknown_token")
    if not isinstance(unknown_token, str):
        raise ValueError("checkpoint tokenizer.unknown_token must be a string")
    return CharacterTokenizer(
        token_to_id={
            str(token): _int_value(index, f"tokenizer.token_to_id.{token}")
            for token, index in token_to_id.items()
        },
        unknown_token=unknown_token,
    )


def save_transformer_checkpoint(
    path: str | Path,
    model: TransformerLanguageModel,
    tokenizer: CharacterTokenizer,
    *,
    step: int,
    metrics: dict[str, float],
    optimizer_state: object | None = None,
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
) -> tuple[TransformerLanguageModel, CharacterTokenizer, CheckpointMetadata]:
    payload = _mapping(
        torch.load(Path(path), map_location=map_location, weights_only=False),
        "root",
    )
    config = _config_from_payload(payload.get("config"))
    model = TransformerLanguageModel(config)
    model.load_state_dict(dict(_mapping(payload.get("model_state"), "model_state")))
    tokenizer = _tokenizer_from_payload(payload.get("tokenizer"))
    metadata = CheckpointMetadata(
        step=_int_value(payload.get("step"), "step"),
        metrics=_float_metrics(payload.get("metrics")),
        optimizer_state=payload.get("optimizer_state"),
    )
    return model, tokenizer, metadata
