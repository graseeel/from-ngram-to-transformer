from pathlib import Path

import torch

from ngram_transformer.config import TransformerConfig
from ngram_transformer.data.tokenizer import CharacterTokenizer
from ngram_transformer.ml.checkpoints import (
    load_transformer_checkpoint,
    save_transformer_checkpoint,
)
from ngram_transformer.ml.transformer import TransformerLanguageModel


def test_transformer_checkpoint_roundtrip(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer.from_texts(["abc"])
    config = TransformerConfig(
        block_size=4,
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
        vocab_size=tokenizer.vocab_size,
    )
    model = TransformerLanguageModel(config)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "checkpoint.pt"

    save_transformer_checkpoint(
        path,
        model,
        tokenizer,
        step=3,
        metrics={"validation_loss": 1.2},
        optimizer_state=optimizer.state_dict(),
    )
    loaded_model, loaded_tokenizer, metadata = load_transformer_checkpoint(path)

    assert loaded_model.config.vocab_size == tokenizer.vocab_size
    assert loaded_tokenizer.decode(tokenizer.encode("abc")) == "abc"
    assert metadata.step == 3
    assert metadata.optimizer_state is not None
