from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ConfigSection = dict[str, object]


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


def _section(raw: ConfigSection, name: str) -> ConfigSection:
    value = raw.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"config section {name!r} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _int_value(section: ConfigSection, key: str, default: int) -> int:
    value = section.get(key, default)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"config field {key!r} must be an integer")


def _optional_int_value(section: ConfigSection, key: str, default: int | None) -> int | None:
    value = section.get(key, default)
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"config field {key!r} must be an integer or null")


def _float_value(section: ConfigSection, key: str, default: float) -> float:
    value = section.get(key, default)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"config field {key!r} must be numeric")


def _optional_float_value(section: ConfigSection, key: str, default: float | None) -> float | None:
    value = section.get(key, default)
    if value is None:
        return None
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"config field {key!r} must be numeric or null")


def _path_value(section: ConfigSection, key: str, default: Path) -> Path:
    value = section.get(key, default)
    if isinstance(value, Path):
        return value
    if isinstance(value, str):
        return Path(value)
    raise ValueError(f"config field {key!r} must be a path string")


def _str_value(section: ConfigSection, key: str, default: str) -> str:
    value = section.get(key, default)
    if isinstance(value, str):
        return value
    raise ValueError(f"config field {key!r} must be a string")


def _data_config(section: ConfigSection) -> DataConfig:
    defaults = DataConfig()
    return DataConfig(
        corpus_path=_path_value(section, "corpus_path", defaults.corpus_path),
        metadata_path=_path_value(section, "metadata_path", defaults.metadata_path),
        train_ratio=_float_value(section, "train_ratio", defaults.train_ratio),
        validation_ratio=_float_value(section, "validation_ratio", defaults.validation_ratio),
        test_ratio=_float_value(section, "test_ratio", defaults.test_ratio),
        block_size=_int_value(section, "block_size", defaults.block_size),
        seed=_int_value(section, "seed", defaults.seed),
    )


def _ngram_config(section: ConfigSection) -> NGramConfig:
    defaults = NGramConfig()
    return NGramConfig(
        n=_int_value(section, "n", defaults.n),
        add_k=_float_value(section, "add_k", defaults.add_k),
    )


def _transformer_config(section: ConfigSection) -> TransformerConfig:
    defaults = TransformerConfig()
    return TransformerConfig(
        block_size=_int_value(section, "block_size", defaults.block_size),
        n_layer=_int_value(section, "n_layer", defaults.n_layer),
        n_head=_int_value(section, "n_head", defaults.n_head),
        n_embd=_int_value(section, "n_embd", defaults.n_embd),
        dropout=_float_value(section, "dropout", defaults.dropout),
        vocab_size=_optional_int_value(section, "vocab_size", defaults.vocab_size),
    )


def _training_config(section: ConfigSection) -> TrainingConfig:
    defaults = TrainingConfig()
    return TrainingConfig(
        batch_size=_int_value(section, "batch_size", defaults.batch_size),
        max_steps=_int_value(section, "max_steps", defaults.max_steps),
        learning_rate=_float_value(section, "learning_rate", defaults.learning_rate),
        eval_interval=_int_value(section, "eval_interval", defaults.eval_interval),
        grad_clip=_float_value(section, "grad_clip", defaults.grad_clip),
        device=_str_value(section, "device", defaults.device),
    )


def _generation_config(section: ConfigSection) -> GenerationConfig:
    defaults = GenerationConfig()
    return GenerationConfig(
        max_new_tokens=_int_value(section, "max_new_tokens", defaults.max_new_tokens),
        temperature=_float_value(section, "temperature", defaults.temperature),
        top_k=_optional_int_value(section, "top_k", defaults.top_k),
        top_p=_optional_float_value(section, "top_p", defaults.top_p),
    )


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    loaded: object = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError("top-level config must be a mapping")
    raw = {str(key): value for key, value in loaded.items()}
    return ProjectConfig(
        data=_data_config(_section(raw, "data")),
        ngram=_ngram_config(_section(raw, "ngram")),
        transformer=_transformer_config(_section(raw, "transformer")),
        training=_training_config(_section(raw, "training")),
        generation=_generation_config(_section(raw, "generation")),
    )
