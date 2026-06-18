import pytest

from ngram_transformer.config import GenerationConfig, TransformerConfig


def test_generation_config_rejects_invalid_top_p() -> None:
    with pytest.raises(ValueError):
        GenerationConfig(top_p=1.5)


def test_transformer_config_requires_divisible_heads() -> None:
    with pytest.raises(ValueError):
        TransformerConfig(n_embd=10, n_head=3)
