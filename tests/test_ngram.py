from ngram_transformer.config import NGramConfig
from ngram_transformer.ml.ngram import NGramLanguageModel


def test_ngram_scores_known_and_unknown_contexts() -> None:
    model = NGramLanguageModel(NGramConfig(n=2, add_k=0.5), vocab_size=2)
    model.fit([0, 1, 0, 1, 0])

    known = model.probability(0, (1,))
    unknown = model.probability(0, (99,))

    assert known > 0
    assert unknown > 0
    assert model.perplexity([0, 1, 0]) > 0


def test_ngram_generation_is_seeded() -> None:
    model = NGramLanguageModel(NGramConfig(n=3, add_k=0.1), vocab_size=3)
    model.fit([0, 1, 2, 0, 1, 2, 0])

    first = model.generate([0, 1], 8, seed=123, temperature=0.8)
    second = model.generate([0, 1], 8, seed=123, temperature=0.8)

    assert first == second
    assert len(first) == 10
