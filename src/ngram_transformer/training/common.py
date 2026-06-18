from __future__ import annotations

from dataclasses import dataclass

from ngram_transformer.config import DataConfig
from ngram_transformer.data.dataset import (
    TextSplits,
    load_corpus_metadata,
    read_text_file,
    split_text,
)
from ngram_transformer.data.tokenizer import CharacterTokenizer


@dataclass(frozen=True)
class PreparedCorpus:
    text: str
    splits: TextSplits
    tokenizer: CharacterTokenizer
    train_ids: list[int]
    validation_ids: list[int]
    test_ids: list[int]
    metadata: dict[str, object]


def prepare_corpus(config: DataConfig) -> PreparedCorpus:
    text = read_text_file(config.corpus_path)
    splits = split_text(
        text,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
    )
    tokenizer = CharacterTokenizer.from_texts([splits.train])
    metadata = load_corpus_metadata(config.metadata_path)
    return PreparedCorpus(
        text=text,
        splits=splits,
        tokenizer=tokenizer,
        train_ids=tokenizer.encode(splits.train),
        validation_ids=tokenizer.encode(splits.validation),
        test_ids=tokenizer.encode(splits.test),
        metadata=metadata,
    )
