"""Token counting for size-based chunking.

Chunk sizes are specified in tokens, so the counter has to be explicit and
deterministic. ``cl100k_base`` is the default because it is the de-facto
standard for "150 tokens" style configurations.

Important measured caveat for this project: cl100k is far less efficient on
Arabic than on English.

    Arabic   1.44 chars/token  ->  150 tokens ~=  216 characters
    English  5.36 chars/token  ->  150 tokens ~=  803 characters

A token budget tuned on English therefore yields Arabic chunks roughly a
quarter of the size. ``CHUNK_UNIT=chars`` counts characters instead, which
treats both scripts equally.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

    def split(self, text: str) -> list[str]:
        """Split into counting units, so a window can be built from them."""
        ...


class CharCounter:
    """Counts characters. Script-neutral."""

    name = "chars"

    def count(self, text: str) -> int:
        return len(text)

    def split(self, text: str) -> list[str]:
        return list(text)

    def join(self, units: list[str]) -> str:
        return "".join(units)


class WordCounter:
    """Counts whitespace-delimited words. Cheap and script-neutral."""

    name = "words"

    def count(self, text: str) -> int:
        return len(text.split())

    def split(self, text: str) -> list[str]:
        return text.split()

    def join(self, units: list[str]) -> str:
        return " ".join(units)


class TiktokenCounter:
    """Counts BPE tokens with tiktoken."""

    def __init__(self, encoding: str = "cl100k_base") -> None:
        import tiktoken

        self.name = f"tokens:{encoding}"
        self._enc = tiktoken.get_encoding(encoding)

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def split(self, text: str) -> list[str]:
        # Decode each token back to its own string so a window of N tokens can
        # be reassembled losslessly.
        return [self._enc.decode([t]) for t in self._enc.encode(text)]

    def join(self, units: list[str]) -> str:
        return "".join(units)


@lru_cache
def get_token_counter(unit: str = "tokens", encoding: str = "cl100k_base"):
    """Build the configured counter. Falls back to words if tiktoken is absent."""
    unit = (unit or "tokens").lower()
    if unit == "chars":
        return CharCounter()
    if unit == "words":
        return WordCounter()
    try:
        return TiktokenCounter(encoding)
    except Exception:  # pragma: no cover - tiktoken is a declared dependency
        return WordCounter()
