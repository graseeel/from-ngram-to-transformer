from pathlib import Path

from ngram_transformer.data.tokenizer import CharacterTokenizer


def test_character_tokenizer_encode_decode_persist(tmp_path: Path) -> None:
    tokenizer = CharacterTokenizer.from_texts(["abba"])
    encoded = tokenizer.encode("abba")
    assert tokenizer.decode(encoded) == "abba"

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    loaded = CharacterTokenizer.load(path)
    assert loaded.decode(encoded) == "abba"
    assert loaded.encode("z") == [loaded.unknown_id]
