from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class DataConfig:
    corpus_path: Path = Path("data/corpus/tiny_corpus.txt")
    metadata_path: Path = Path("data/corpus/metadata.json")
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    block_size: int = 64
    seed: int = 1337

    def __post_init__(self) -> None:
        _positive(self.block_size, "block_size")
        total = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError("data split ratios must sum to 1.0")
        if min(self.train_ratio, self.validation_ratio, self.test_ratio) <= 0:
            raise ValueError("data split ratios must be positive")


@dataclass(frozen=True)
class NGramConfig:
    n: int = 3
    add_k: float = 0.1

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be at least 1")
        _positive(self.add_k, "add_k")


@dataclass(frozen=True)
class TransformerConfig:
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    dropout: float = 0.1
    vocab_size: int | None = None

    def __post_init__(self) -> None:
        for name in ("block_size", "n_layer", "n_head", "n_embd"):
            _positive(getattr(self, name), name)
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

    def with_vocab_size(self, vocab_size: int) -> TransformerConfig:
        _positive(vocab_size, "vocab_size")
        return TransformerConfig(
            block_size=self.block_size,
            n_layer=self.n_layer,
            n_head=self.n_head,
            n_embd=self.n_embd,
            dropout=self.dropout,
            vocab_size=vocab_size,
        )


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 16
    max_steps: int = 100
    learning_rate: float = 5e-4
    eval_interval: int = 25
    grad_clip: float = 1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("batch_size", "max_steps", "eval_interval"):
            _positive(getattr(self, name), name)
        _positive(self.learning_rate, "learning_rate")
        _positive(self.grad_clip, "grad_clip")


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 120
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None

    def __post_init__(self) -> None:
        _positive(self.max_new_tokens, "max_new_tokens")
        _positive(self.temperature, "temperature")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when provided")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1] when provided")


@dataclass(frozen=True)
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    ngram: NGramConfig = field(default_factory=NGramConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    return value


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("top-level config must be a mapping")
    return ProjectConfig(
        data=DataConfig(**_section(raw, "data")),
        ngram=NGramConfig(**_section(raw, "ngram")),
        transformer=TransformerConfig(**_section(raw, "transformer")),
        training=TrainingConfig(**_section(raw, "training")),
        generation=GenerationConfig(**_section(raw, "generation")),
    )
