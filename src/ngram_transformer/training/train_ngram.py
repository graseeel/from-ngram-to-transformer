from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ngram_transformer.config import load_config
from ngram_transformer.ml.ngram import NGramLanguageModel
from ngram_transformer.training.common import prepare_corpus


def train(config_path: Path, output_dir: Path) -> dict[str, float]:
    config = load_config(config_path)
    prepared = prepare_corpus(config.data)
    model = NGramLanguageModel(config.ngram, prepared.tokenizer.vocab_size)
    model.fit(prepared.train_ids)
    validation_metrics = model.evaluate(prepared.validation_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "ngram_model.json")
    prepared.tokenizer.save(output_dir / "tokenizer.json")

    metrics = {
        "validation_log_likelihood": validation_metrics.log_likelihood,
        "validation_perplexity": validation_metrics.perplexity,
        "vocab_size": float(prepared.tokenizer.vocab_size),
    }
    report = {
        "model": "ngram",
        "config": {"ngram": asdict(config.ngram), "data": asdict(config.data)},
        "metrics": metrics,
        "corpus": prepared.metadata,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the add-k smoothed N-gram model.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ngram"))
    args = parser.parse_args()
    metrics = train(args.config, args.output_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
