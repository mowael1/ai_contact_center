"""Small dependency-free lexical ranking helpers for hybrid retrieval."""

from __future__ import annotations

import re
import unicodedata
from typing import Sequence

_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u0640]")


def normalize_tokens(text: str) -> set[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _DIACRITICS_RE.sub("", text)
    text = text.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}))
    tokens = set(_TOKEN_RE.findall(text))
    expanded = set(tokens)
    for token in tokens:
        if token.startswith("ال") and len(token) > 3:
            expanded.add(token[2:])
    return expanded


def lexical_score(query_terms: Sequence[str], text: str) -> float:
    terms = {term for term in query_terms if len(term) > 1}
    if not terms:
        return 0.0
    content_terms = normalize_tokens(text)
    overlap = terms & content_terms
    number_hits = sum(term.isdigit() for term in overlap)
    return len(overlap) / len(terms) + 0.75 * number_hits
