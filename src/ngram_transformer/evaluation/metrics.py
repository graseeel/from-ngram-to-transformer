from __future__ import annotations

import math
from dataclasses import dataclass

import torch.nn as nn


def perplexity_from_loss(loss: float) -> float:
    if loss < 0:
        raise ValueError("loss cannot be negative")
    return math.exp(loss)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


@dataclass(frozen=True)
class GenerationComparison:
    prompt: str
    ngram_text: str
    transformer_text: str | None
