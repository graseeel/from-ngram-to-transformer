from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ngram_transformer.config import ProjectConfig, TransformerConfig, load_config
from ngram_transformer.data.dataset import TokenSequenceDataset
from ngram_transformer.data.tokenizer import CharacterTokenizer
from ngram_transformer.evaluation.metrics import count_parameters, perplexity_from_loss
from ngram_transformer.ml.checkpoints import (
    load_transformer_checkpoint,
    save_transformer_checkpoint,
)
from ngram_transformer.ml.transformer import TransformerLanguageModel
from ngram_transformer.training.common import PreparedCorpus, prepare_corpus

TransformerLoader = DataLoader[tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True)
class TransformerDataLoaders:
    train: TransformerLoader
    validation: TransformerLoader


@dataclass(frozen=True)
class TrainingState:
    model: TransformerLanguageModel
    tokenizer: CharacterTokenizer
    start_step: int
    optimizer_state: object | None


def _average_loss(
    model: TransformerLanguageModel,
    dataloader: TransformerLoader,
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


def _build_dataloaders(config: ProjectConfig, prepared: PreparedCorpus) -> TransformerDataLoaders:
    train_dataset = TokenSequenceDataset(prepared.train_ids, config.transformer.block_size)
    validation_dataset = TokenSequenceDataset(
        prepared.validation_ids,
        config.transformer.block_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.training.batch_size)
    return TransformerDataLoaders(train=train_loader, validation=validation_loader)


def _expected_model_config(config: ProjectConfig, prepared: PreparedCorpus) -> TransformerConfig:
    return config.transformer.with_vocab_size(prepared.tokenizer.vocab_size)


def _validate_resume_checkpoint(
    model: TransformerLanguageModel,
    tokenizer: CharacterTokenizer,
    expected_config: TransformerConfig,
    expected_tokenizer: CharacterTokenizer,
) -> None:
    if model.config != expected_config:
        raise ValueError("checkpoint transformer config does not match the training config")
    if tokenizer.token_to_id != expected_tokenizer.token_to_id:
        raise ValueError("checkpoint tokenizer does not match the prepared training corpus")


def _load_or_create_state(
    config: ProjectConfig,
    prepared: PreparedCorpus,
    resume: Path | None,
    device: torch.device,
) -> TrainingState:
    expected_config = _expected_model_config(config, prepared)
    if resume is not None:
        model, tokenizer, metadata = load_transformer_checkpoint(resume, map_location=device)
        _validate_resume_checkpoint(model, tokenizer, expected_config, prepared.tokenizer)
        return TrainingState(
            model=model,
            tokenizer=tokenizer,
            start_step=metadata.step,
            optimizer_state=metadata.optimizer_state,
        )
    return TrainingState(
        model=TransformerLanguageModel(expected_config),
        tokenizer=prepared.tokenizer,
        start_step=0,
        optimizer_state=None,
    )


def _metrics(
    model: TransformerLanguageModel,
    train_loss: float,
    validation_loss: float,
    started: float,
) -> dict[str, float]:
    return {
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "validation_perplexity": perplexity_from_loss(validation_loss),
        "parameter_count": float(count_parameters(model)),
        "training_seconds": time.perf_counter() - started,
    }


def _write_report(
    output_dir: Path,
    model: TransformerLanguageModel,
    config: ProjectConfig,
    prepared: PreparedCorpus,
    metrics: dict[str, float],
) -> None:
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


def _load_optimizer_state(optimizer: torch.optim.Optimizer, state: object | None) -> None:
    if state is None:
        return
    if not isinstance(state, dict):
        raise ValueError("checkpoint optimizer_state must be a dictionary")
    optimizer.load_state_dict(state)


def _save_checkpoint(
    output_dir: Path,
    state: TrainingState,
    step: int,
    metrics: dict[str, float],
    optimizer: torch.optim.Optimizer,
) -> None:
    save_transformer_checkpoint(
        output_dir / "transformer_checkpoint.pt",
        state.model,
        state.tokenizer,
        step=step,
        metrics=metrics,
        optimizer_state=optimizer.state_dict(),
    )


def _next_batch(
    iterator: Iterator[tuple[torch.Tensor, torch.Tensor]],
    loader: TransformerLoader,
) -> tuple[torch.Tensor, torch.Tensor, Iterator[tuple[torch.Tensor, torch.Tensor]]]:
    try:
        x, y = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        x, y = next(iterator)
    return x, y, iterator


def train(config_path: Path, output_dir: Path, resume: Path | None = None) -> dict[str, float]:
    config = load_config(config_path)
    prepared = prepare_corpus(config.data)
    device = torch.device(config.training.device)
    loaders = _build_dataloaders(config, prepared)
    state = _load_or_create_state(config, prepared, resume, device)

    state.model.to(device)
    optimizer = torch.optim.AdamW(state.model.parameters(), lr=config.training.learning_rate)
    _load_optimizer_state(optimizer, state.optimizer_state)

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    last_loss = float("nan")
    step = state.start_step
    iterator: Iterator[tuple[torch.Tensor, torch.Tensor]] = iter(loaders.train)
    while step < config.training.max_steps:
        x, y, iterator = _next_batch(iterator, loaders.train)
        optimizer.zero_grad(set_to_none=True)
        _, loss = state.model(x.to(device), y.to(device))
        if loss is None:
            raise RuntimeError("loss was not computed")
        loss.backward()
        # Clipping keeps tiny demo runs stable when users swap in noisier corpora.
        torch.nn.utils.clip_grad_norm_(state.model.parameters(), max_norm=config.training.grad_clip)
        optimizer.step()
        last_loss = float(loss.item())
        step += 1

        if step % config.training.eval_interval == 0 or step == config.training.max_steps:
            validation_loss = _average_loss(state.model, loaders.validation, device)
            _save_checkpoint(
                output_dir,
                state,
                step,
                _metrics(state.model, last_loss, validation_loss, started),
                optimizer,
            )

    final_validation_loss = _average_loss(state.model, loaders.validation, device)
    metrics = _metrics(state.model, last_loss, final_validation_loss, started)
    _write_report(output_dir, state.model, config, prepared, metrics)
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
