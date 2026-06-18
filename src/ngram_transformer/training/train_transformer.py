from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ngram_transformer.config import load_config
from ngram_transformer.data.dataset import TokenSequenceDataset
from ngram_transformer.evaluation.metrics import count_parameters, perplexity_from_loss
from ngram_transformer.ml.checkpoints import (
    load_transformer_checkpoint,
    save_transformer_checkpoint,
)
from ngram_transformer.ml.transformer import TransformerLanguageModel
from ngram_transformer.training.common import prepare_corpus


def _average_loss(
    model: TransformerLanguageModel,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    max_batches: int = 10,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for batch_index, (x, y) in enumerate(dataloader):
            if batch_index >= max_batches:
                break
            _, loss = model(x.to(device), y.to(device))
            if loss is None:
                raise RuntimeError("loss was not computed")
            losses.append(float(loss.item()))
    model.train()
    if not losses:
        raise ValueError("cannot evaluate loss on an empty dataloader")
    return sum(losses) / len(losses)


def train(config_path: Path, output_dir: Path, resume: Path | None = None) -> dict[str, float]:
    config = load_config(config_path)
    prepared = prepare_corpus(config.data)
    device = torch.device(config.training.device)

    train_dataset = TokenSequenceDataset(prepared.train_ids, config.transformer.block_size)
    validation_dataset = TokenSequenceDataset(
        prepared.validation_ids,
        config.transformer.block_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.training.batch_size)

    if resume is not None:
        model, tokenizer, checkpoint_meta = load_transformer_checkpoint(resume, map_location=device)
        start_step = int(checkpoint_meta["step"])
        prepared_tokenizer = tokenizer
    else:
        model_config = config.transformer.with_vocab_size(prepared.tokenizer.vocab_size)
        model = TransformerLanguageModel(model_config)
        start_step = 0
        checkpoint_meta = {"optimizer_state": None}
        prepared_tokenizer = prepared.tokenizer

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    if checkpoint_meta.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint_meta["optimizer_state"])

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    last_loss = float("nan")
    step = start_step
    iterator = iter(train_loader)
    while step < config.training.max_steps:
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x.to(device), y.to(device))
        if loss is None:
            raise RuntimeError("loss was not computed")
        loss.backward()
        # Clipping keeps tiny demo runs stable when users swap in noisier corpora.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.grad_clip)
        optimizer.step()
        last_loss = float(loss.item())
        step += 1

        if step % config.training.eval_interval == 0 or step == config.training.max_steps:
            validation_loss = _average_loss(model, validation_loader, device)
            metrics = {
                "train_loss": last_loss,
                "validation_loss": validation_loss,
                "validation_perplexity": perplexity_from_loss(validation_loss),
                "parameter_count": float(count_parameters(model)),
                "training_seconds": time.perf_counter() - started,
            }
            save_transformer_checkpoint(
                output_dir / "transformer_checkpoint.pt",
                model,
                prepared_tokenizer,
                step=step,
                metrics=metrics,
                optimizer_state=optimizer.state_dict(),
            )

    final_validation_loss = _average_loss(model, validation_loader, device)
    metrics = {
        "train_loss": last_loss,
        "validation_loss": final_validation_loss,
        "validation_perplexity": perplexity_from_loss(final_validation_loss),
        "parameter_count": float(count_parameters(model)),
        "training_seconds": time.perf_counter() - started,
    }
    report = {
        "model": "transformer",
        "config": {"transformer": asdict(model.config), "training": asdict(config.training)},
        "metrics": metrics,
        "corpus": prepared.metadata,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the small decoder-only Transformer.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/transformer"))
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    metrics = train(args.config, args.output_dir, args.resume)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
