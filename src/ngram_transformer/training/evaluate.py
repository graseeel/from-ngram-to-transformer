from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ngram_transformer.config import load_config
from ngram_transformer.data.dataset import TokenSequenceDataset
from ngram_transformer.evaluation.metrics import count_parameters, perplexity_from_loss
from ngram_transformer.ml.checkpoints import load_transformer_checkpoint
from ngram_transformer.ml.ngram import NGramLanguageModel
from ngram_transformer.training.common import prepare_corpus


def evaluate(
    config_path: Path,
    ngram_model_path: Path,
    transformer_checkpoint_path: Path | None,
    output_path: Path,
) -> dict[str, object]:
    config = load_config(config_path)
    prepared = prepare_corpus(config.data)
    ngram = NGramLanguageModel.load(ngram_model_path)
    ngram_metrics = ngram.evaluate(prepared.test_ids)

    report: dict[str, object] = {
        "corpus": prepared.metadata,
        "ngram": {
            "test_log_likelihood": ngram_metrics.log_likelihood,
            "test_perplexity": ngram_metrics.perplexity,
        },
    }

    if transformer_checkpoint_path is not None and transformer_checkpoint_path.exists():
        model, tokenizer, metadata = load_transformer_checkpoint(transformer_checkpoint_path)
        test_ids = tokenizer.encode(prepared.splits.test)
        test_dataset = TokenSequenceDataset(test_ids, model.config.block_size)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)
        losses: list[float] = []
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                _, loss = model(x, y)
                if loss is None:
                    raise RuntimeError("loss was not computed")
                losses.append(float(loss.item()))
        mean_loss = sum(losses) / len(losses)
        report["transformer"] = {
            "test_loss": mean_loss,
            "test_perplexity": perplexity_from_loss(mean_loss),
            "parameter_count": count_parameters(model),
            "checkpoint_step": metadata["step"],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on the held-out test split.",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--ngram-model",
        type=Path,
        default=Path("artifacts/ngram/ngram_model.json"),
    )
    parser.add_argument(
        "--transformer-checkpoint",
        type=Path,
        default=Path("artifacts/transformer/transformer_checkpoint.pt"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/report.json"))
    args = parser.parse_args()
    checkpoint = args.transformer_checkpoint if args.transformer_checkpoint.exists() else None
    report = evaluate(args.config, args.ngram_model, checkpoint, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
