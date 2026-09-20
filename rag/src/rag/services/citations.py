"""Citation construction.

Citations are built from retrieved metadata only. The LLM supplies bracketed
indices; anything it emits that does not map to a retrieved passage is
discarded, which is what makes a fabricated citation structurally impossible.
"""

from __future__ import annotations

import re
from typing import Sequence

from rag.models import Citation

_CITE = re.compile(r"\[(\d{1,2})\]")


def extract_cited_indices(answer: str) -> list[int]:
    """Bracketed numbers the model used, in first-appearance order."""
    seen: list[int] = []
    for match in _CITE.finditer(answer or ""):
        index = int(match.group(1))
        if index not in seen:
            seen.append(index)
    return seen


def resolve_citations(
    answer: str, available: Sequence[Citation], keep_all: bool = False
) -> list[Citation]:
    """Map the model's indices onto real citations, dropping invalid ones."""
    by_index = {c.index: c for c in available}
    cited = extract_cited_indices(answer)
    resolved = [by_index[i] for i in cited if i in by_index]
    if resolved:
        return resolved
    # The model answered without citing: fall back to the passages it was
    # given rather than presenting an uncited answer.
    return list(available) if keep_all else []


def strip_invalid_markers(answer: str, valid_indices: set[int]) -> str:
    """Remove bracketed references that point at passages that do not exist."""

    def repl(match: re.Match) -> str:
        # Keep a valid marker exactly as written; drop an invalid one along with
        # the whitespace that preceded it, so no double space is left behind.
        return match.group(0) if int(match.group(2)) in valid_indices else ""

    cleaned = re.sub(r"(\s*)\[(\d{1,2})\]", repl, answer or "")
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()
