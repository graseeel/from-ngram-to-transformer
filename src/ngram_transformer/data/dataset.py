from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def normalize_text(text: str) -> str:
    """Normalize line endings while preserving punctuation and casing."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip() + "\n"


def read_text_file(path: str | Path) -> str:
    source = Path(path)
    if source.suffix != ".txt":
        raise ValueError("corpus files must use the .txt extension")
    if not source.exists():
        raise FileNotFoundError(source)
    return normalize_text(source.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class TextSplits:
    train: str
    validation: str
    test: str


def split_text(
    text: str,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> TextSplits:
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1.0")
    train_end = int(len(text) * train_ratio)
    validation_end = train_end + int(len(text) * validation_ratio)
    return TextSplits(
        train=text[:train_end],
        validation=text[train_end:validation_end],
        test=text[validation_end:],
    )


def load_corpus_metadata(path: str | Path) -> dict[str, Any]:
    metadata_path = Path(path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {"name", "origin", "license"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"corpus metadata missing required keys: {sorted(missing)}")
    return payload


class TokenSequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Contiguous token windows for autoregressive next-token training."""

    def __init__(self, token_ids: list[int], block_size: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if len(token_ids) <= block_size:
            raise ValueError("token_ids must be longer than block_size")
        self._tokens = torch.tensor(token_ids, dtype=torch.long)
        self._block_size = block_size

    def __len__(self) -> int:
        return len(self._tokens) - self._block_size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        chunk = self._tokens[index : index + self._block_size + 1]
        return chunk[:-1], chunk[1:]
