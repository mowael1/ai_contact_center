"""Deterministic sentence segmentation for Arabic, English and mixed text.

No model, no downloads. The splitter walks the string once and only breaks at
a terminator that is followed by whitespace and a plausible sentence start,
after masking the constructs that most often cause false splits in this
corpus: URLs, USSD codes (``*880#``), decimals, prices and abbreviations.
"""

from __future__ import annotations

import re

# Sentence terminators: Latin plus Arabic question mark and full stop.
TERMINATORS = ".!?؟۔⁉‽"

ABBREVIATIONS = {
    # English
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g",
    "i.e", "inc", "ltd", "co", "vol", "fig", "approx", "dept", "eg", "ie",
    # Units and currencies are deliberately NOT listed: "costs 70.50 EGP."
    # legitimately ends a sentence, and the decimal mask already protects
    # numbers like "1.5 GB" from being split.
    # Arabic
    "الد", "أ.د", "ص", "م", "هـ",
}

_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_USSD = re.compile(r"[*#][\d*#]{2,}#?")          # *880#, *9*7#
_DECIMAL = re.compile(r"\d+(?:[.,]\d+)+")         # 1.5, 1,000.50
_ELLIPSIS = re.compile(r"\.{2,}|…")
_ORDINAL = re.compile(r"\b\d+\.(?=\s+[a-z؀-ۿ])")  # "1. item" in a list

_PLACEHOLDER = ""  # private-use area, cannot occur in real content

_BULLET_START = re.compile(r"^\s*(?:[-•*●•]|\d+[.)]|[a-z][.)])\s+", re.I)


class _Mask:
    """Replaces protected spans with placeholders, then restores them."""

    def __init__(self) -> None:
        self.store: list[str] = []

    def mask(self, text: str) -> str:
        def repl(match: re.Match) -> str:
            self.store.append(match.group(0))
            return f"{_PLACEHOLDER}{len(self.store) - 1}{_PLACEHOLDER}"

        for pattern in (_URL, _EMAIL, _USSD, _DECIMAL, _ELLIPSIS, _ORDINAL):
            text = pattern.sub(repl, text)
        return text

    def unmask(self, text: str) -> str:
        def repl(match: re.Match) -> str:
            return self.store[int(match.group(1))]

        return re.sub(f"{_PLACEHOLDER}(\\d+){_PLACEHOLDER}", repl, text)


def _ends_with_abbreviation(fragment: str) -> bool:
    match = re.search(r"([A-Za-z؀-ۿ.]+)\.$", fragment.strip())
    if not match:
        return False
    return match.group(1).lower().strip(".") in ABBREVIATIONS


def split_sentences(text: str, min_chars: int = 2) -> list[str]:
    """Split ``text`` into sentences, preserving original substrings.

    Newlines are treated as hard boundaries because section text from the HTML
    extractor puts each list item and table row on its own line.
    """
    if not text or not text.strip():
        return []

    sentences: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        sentences.extend(_split_line(line))

    merged = _merge_short(sentences, min_chars)
    return [s for s in merged if s.strip()]


def _split_line(line: str) -> list[str]:
    masker = _Mask()
    masked = masker.mask(line)

    out: list[str] = []
    start = 0
    i = 0
    while i < len(masked):
        char = masked[i]
        if char in TERMINATORS:
            # Consume a run of terminators ("?!", "؟؟").
            j = i + 1
            while j < len(masked) and masked[j] in TERMINATORS:
                j += 1
            rest = masked[j:]
            if rest and not rest[0].isspace():
                i = j
                continue
            candidate = masked[start:j]
            if _ends_with_abbreviation(candidate):
                i = j
                continue
            out.append(candidate.strip())
            start = j
            i = j
            continue
        i += 1
    tail = masked[start:].strip()
    if tail:
        out.append(tail)
    return [masker.unmask(s) for s in out if s.strip()]


def _merge_short(sentences: list[str], min_chars: int) -> list[str]:
    """Glue fragments that are too short to stand alone onto their neighbour.

    Bullet lines are never merged forward into a preceding prose sentence,
    so list structure survives.
    """
    out: list[str] = []
    for sentence in sentences:
        if (
            out
            and len(sentence) < min_chars
            and not _BULLET_START.match(sentence)
        ):
            out[-1] = f"{out[-1]} {sentence}".strip()
        else:
            out.append(sentence)
    return out
