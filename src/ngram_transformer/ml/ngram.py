from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ngram_transformer.config import NGramConfig

START_TOKEN_ID = -1


@dataclass(frozen=True)
class NGramMetrics:
    log_likelihood: float
    perplexity: float


class NGramLanguageModel:
    """Add-k smoothed N-gram language model over integer token ids."""

    def __init__(self, config: NGramConfig, vocab_size: int) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.config = config
        self.vocab_size = vocab_size
        self.context_counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.unigram_counts: Counter[int] = Counter()
        self._is_fit = False

    @property
    def context_width(self) -> int:
        return max(self.config.n - 1, 0)

    def fit(self, token_ids: list[int]) -> None:
        if not token_ids:
            raise ValueError("token_ids cannot be empty")
        padded = [START_TOKEN_ID] * self.context_width + token_ids
        for index, token_id in enumerate(token_ids):
            context_start = index
            context_end = index + self.context_width
            context = tuple(padded[context_start:context_end])
            self.context_counts[context][token_id] += 1
            self.unigram_counts[token_id] += 1
        self._is_fit = True

    def _distribution_counts(self, context: tuple[int, ...]) -> Counter[int]:
        if self.context_width == 0:
            return self.unigram_counts
        normalized = self._normalize_context(context)
        counts = self.context_counts.get(normalized)
        return counts if counts else self.unigram_counts

    def _normalize_context(self, context: tuple[int, ...]) -> tuple[int, ...]:
        if self.context_width == 0:
            return ()
        if len(context) >= self.context_width:
            return tuple(context[-self.context_width :])
        return tuple([START_TOKEN_ID] * (self.context_width - len(context)) + list(context))

    def probability(self, token_id: int, context: tuple[int, ...]) -> float:
        if not self._is_fit:
            raise RuntimeError("model must be fit before scoring")
        counts = self._distribution_counts(context)
        numerator = counts[token_id] + self.config.add_k
        denominator = sum(counts.values()) + self.config.add_k * self.vocab_size
        return numerator / denominator

    def log_likelihood(self, token_ids: list[int]) -> float:
        if not token_ids:
            raise ValueError("token_ids cannot be empty")
        padded = [START_TOKEN_ID] * self.context_width + token_ids
        total = 0.0
        for index, token_id in enumerate(token_ids):
            context = tuple(padded[index : index + self.context_width])
            total += math.log(self.probability(token_id, context))
        return total

    def perplexity(self, token_ids: list[int]) -> float:
        return math.exp(-self.log_likelihood(token_ids) / len(token_ids))

    def evaluate(self, token_ids: list[int]) -> NGramMetrics:
        ll = self.log_likelihood(token_ids)
        return NGramMetrics(log_likelihood=ll, perplexity=math.exp(-ll / len(token_ids)))

    def next_token_distribution(self, context: tuple[int, ...], temperature: float) -> list[float]:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        probabilities = [self.probability(token_id, context) for token_id in range(self.vocab_size)]
        if temperature != 1.0:
            probabilities = [probability ** (1.0 / temperature) for probability in probabilities]
        total = sum(probabilities)
        return [probability / total for probability in probabilities]

    def generate(
        self,
        seed_ids: list[int],
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        greedy: bool = False,
    ) -> list[int]:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        rng = random.Random(seed)
        output = list(seed_ids)
        for _ in range(max_new_tokens):
            context = tuple(output[-self.context_width :]) if self.context_width else ()
            probabilities = self.next_token_distribution(context, temperature)
            next_id = self._choose(probabilities, rng, top_k=top_k, top_p=top_p, greedy=greedy)
            output.append(next_id)
        return output

    def _choose(
        self,
        probabilities: list[float],
        rng: random.Random,
        *,
        top_k: int | None,
        top_p: float | None,
        greedy: bool,
    ) -> int:
        ranked = sorted(enumerate(probabilities), key=lambda item: item[1], reverse=True)
        if top_k is not None:
            if top_k <= 0:
                raise ValueError("top_k must be positive when provided")
            ranked = ranked[:top_k]
        if top_p is not None:
            if not 0 < top_p <= 1:
                raise ValueError("top_p must be in (0, 1] when provided")
            kept: list[tuple[int, float]] = []
            running = 0.0
            for token_id, probability in ranked:
                kept.append((token_id, probability))
                running += probability
                if running >= top_p:
                    break
            ranked = kept
        if greedy:
            return ranked[0][0]
        total = sum(probability for _, probability in ranked)
        draw = rng.random() * total
        running = 0.0
        for token_id, probability in ranked:
            running += probability
            if running >= draw:
                return token_id
        return ranked[-1][0]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "vocab_size": self.vocab_size,
            "context_counts": [
                {"context": list(context), "counts": dict(counter)}
                for context, counter in self.context_counts.items()
            ],
            "unigram_counts": dict(self.unigram_counts),
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> NGramLanguageModel:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(NGramConfig(**payload["config"]), vocab_size=int(payload["vocab_size"]))
        for item in payload["context_counts"]:
            context = tuple(int(value) for value in item["context"])
            model.context_counts[context] = Counter(
                {int(token_id): int(count) for token_id, count in item["counts"].items()}
            )
        model.unigram_counts = Counter(
            {int(token_id): int(count) for token_id, count in payload["unigram_counts"].items()}
        )
        model._is_fit = True
        return model
