"""Detection and repair of reversed Arabic text from PDF extraction.

Some PDF producers write Arabic runs in visual order rather than logical
order. The extractor then returns each line character-reversed:

    extracted:  عاجرتسالاو لادبتسالا ةسايس
    correct  :  سياسة الاستبدال والاسترجاع

This passes every naive quality check - the script is Arabic, the characters
are real letters, word lengths look normal - yet the text is unusable: it
embeds to noise and retrieval fails silently. So it is detected explicitly.

**Signal.** Arabic morphology is strongly asymmetric. The definite article
``ال`` is extremely common at the *start* of words and almost never appears as
``لا`` at the *end*. Reversal flips that, along with common function words
(``في``/``يف``, ``من``/``نم``, ``على``/``ىلع``). Comparing the forward and
reversed readings of the same text gives a clear verdict with no model.

**Repair.** The producer reversed characters *and* word order together, so
reversing each line restores both.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ARABIC_WORD = re.compile(r"[؀-ۿ]+")

#: Frequent Arabic function words and their character reversals.
_MARKERS = (
    ("في", "يف"), ("من", "نم"), ("على", "ىلع"), ("إلى", "ىلإ"),
    ("التي", "يتلا"), ("الذي", "يذلا"), ("هذا", "اذه"), ("هذه", "هذه"),
    ("عن", "نع"), ("مع", "عم"), ("أو", "وأ"), ("كل", "لك"),
)


@dataclass(slots=True)
class ReversalVerdict:
    is_reversed: bool
    forward_score: int
    reversed_score: int
    arabic_words: int
    confidence: float

    def to_dict(self) -> dict:
        return {
            "is_reversed": self.is_reversed,
            "forward_score": self.forward_score,
            "reversed_score": self.reversed_score,
            "arabic_words": self.arabic_words,
            "confidence": round(self.confidence, 3),
        }


def _score(text: str) -> int:
    """How much this reading looks like real Arabic."""
    words = ARABIC_WORD.findall(text)
    if not words:
        return 0
    score = sum(1 for w in words if w.startswith("ال") and len(w) > 2)
    for forward, _ in _MARKERS:
        score += text.count(f" {forward} ")
    return score


def detect_reversed(text: str, min_words: int = 8) -> ReversalVerdict:
    """Compare the forward reading against the line-reversed reading."""
    words = ARABIC_WORD.findall(text)
    if len(words) < min_words:
        return ReversalVerdict(False, 0, 0, len(words), 0.0)

    forward = _score(text)
    backward = _score(reverse_lines(text))
    total = forward + backward
    confidence = abs(forward - backward) / total if total else 0.0
    # Require a clear margin: a genuine document scores far higher forward.
    is_reversed = backward > forward and confidence >= 0.5
    return ReversalVerdict(is_reversed, forward, backward, len(words), confidence)


def reverse_lines(text: str) -> str:
    """Reverse each line's characters, preserving line order."""
    return "\n".join(line[::-1] for line in text.split("\n"))


def repair(text: str) -> tuple[str, ReversalVerdict]:
    """Return (possibly repaired text, verdict)."""
    verdict = detect_reversed(text)
    return (reverse_lines(text) if verdict.is_reversed else text), verdict
