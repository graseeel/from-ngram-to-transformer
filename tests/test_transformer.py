import torch

from ngram_transformer.config import TransformerConfig
from ngram_transformer.ml.transformer import TransformerLanguageModel, build_causal_mask


def test_causal_mask_blocks_future_positions() -> None:
    mask = build_causal_mask(4, "cpu")
    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_transformer_forward_shapes_and_loss() -> None:
    config = TransformerConfig(
        block_size=4,
        n_layer=1,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        vocab_size=6,
    )
    model = TransformerLanguageModel(config)
    x = torch.tensor([[0, 1, 2, 3]])
    logits, loss = model(x, x)

    assert logits.shape == (1, 4, 6)
    assert loss is not None
    assert loss.item() > 0


def test_transformer_generation_is_seeded() -> None:
    torch.manual_seed(7)
    config = TransformerConfig(
        block_size=4,
        n_layer=1,
        n_head=2,
        n_embd=8,
        dropout=0.0,
        vocab_size=6,
    )
    model = TransformerLanguageModel(config)

    first = model.generate([0, 1], 5, seed=99)
    second = model.generate([0, 1], 5, seed=99)

    assert first == second
    assert len(first) == 7
