from __future__ import annotations

import torch


def validate_sampling_args(temperature: float, top_k: int | None, top_p: float | None) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be positive when provided")
    if top_p is not None and not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1] when provided")


def filter_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    validate_sampling_args(temperature, top_k, top_p)
    filtered = logits / temperature
    if top_k is not None and top_k < filtered.size(-1):
        values, _ = torch.topk(filtered, top_k)
        threshold = values[..., -1, None]
        filtered = filtered.masked_fill(filtered < threshold, float("-inf"))
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        # The highest-probability token must remain available even for very small top_p.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        original_remove = torch.zeros_like(remove)
        original_remove.scatter_(dim=-1, index=sorted_indices, src=remove)
        filtered = filtered.masked_fill(original_remove, float("-inf"))
    return filtered


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    generator: torch.Generator | None = None,
    greedy: bool = False,
) -> torch.Tensor:
    filtered = filter_logits(logits, temperature=temperature, top_k=top_k, top_p=top_p)
    if greedy:
        return torch.argmax(filtered, dim=-1)
    probabilities = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator).squeeze(-1)
