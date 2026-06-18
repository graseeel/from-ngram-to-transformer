from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CharacterTokenizer:
    """Character tokenizer with a stable unknown token for unseen input."""

    token_to_id: dict[str, int]
    unknown_token: str = "<unk>"

    @classmethod
    def from_texts(cls, texts: list[str], unknown_token: str = "<unk>") -> CharacterTokenizer:
        chars = sorted({char for text in texts for char in text})
        if unknown_token in chars:
            raise ValueError("unknown_token must not appear as a literal corpus character")
        vocab = [unknown_token, *chars]
        return cls({token: index for index, token in enumerate(vocab)}, unknown_token)

    @property
    def id_to_token(self) -> dict[int, str]:
        return {index: token for token, index in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def unknown_id(self) -> int:
        return self.token_to_id[self.unknown_token]

    def encode(self, text: str) -> list[int]:
        return [self.token_to_id.get(char, self.unknown_id) for char in text]

    def decode(self, token_ids: list[int]) -> str:
        id_to_token = self.id_to_token
        return "".join(id_to_token.get(token_id, self.unknown_token) for token_id in token_ids)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "character",
            "unknown_token": self.unknown_token,
            "token_to_id": self.token_to_id,
        }
        target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CharacterTokenizer:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("type") != "character":
            raise ValueError("only character tokenizer payloads are supported")
        token_to_id = payload["token_to_id"]
        if not isinstance(token_to_id, dict):
            raise ValueError("token_to_id must be an object")
        return cls(
            token_to_id={str(token): int(index) for token, index in token_to_id.items()},
            unknown_token=str(payload["unknown_token"]),
        )
