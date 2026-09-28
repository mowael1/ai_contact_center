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
_ARABIC_QUESTION_STARTERS = {"هل", "ازاي", "إزاي", "ايه", "إيه", "ليه", "كيف", "ماذا", "ما", "كل"}

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


def detect_reversed_word_order(text: str) -> bool:
    """Detect PDFs whose Arabic words are emitted in visual (right-to-left) order.

    Character order can be correct while the *word sequence* on a line is
    reversed. FAQ-style question starters provide a conservative signal: in a
    correctly ordered document they begin lines; in a visually ordered PDF
    they tend to appear at line ends. Require several examples and a strong
    directional imbalance before changing any text.
    """
    from rag.text_utils import arabic_ratio

    starts = ends = eligible = 0
    for line in text.splitlines():
        words = line.split()
        if len(words) < 4 or arabic_ratio(line) < 0.65 or "|" in line:
            continue
        eligible += 1
        first = re.sub(r"^[^؀-ۿ]+|[^؀-ۿ]+$", "", words[0])
        last = re.sub(r"^[^؀-ۿ]+|[^؀-ۿ]+$", "", words[-1])
        starts += first in _ARABIC_QUESTION_STARTERS
        ends += last in _ARABIC_QUESTION_STARTERS
    return eligible >= 8 and ends >= 3 and ends >= max(3, starts * 3) and ends / eligible >= 0.15


def repair_pdf_word_order(text: str) -> str:
    """Restore Arabic line and table reading order after PDF extraction."""
    from rag.text_utils import arabic_ratio

    repaired: list[str] = []
    for line in text.splitlines():
        if "|" in line and arabic_ratio(line) >= 0.45:
            # PDF table rows are normalized cell-by-cell in pdf_loader, where
            # we still know which cells came from the table API and which came
            # from adjacent page text.
            repaired.append(line)
        elif len(line.split()) >= 2 and arabic_ratio(line) >= 0.65:
            numeric_prefix = re.match(r"^(\d[\d,./+%]*)\s+(?:[-–]\s*)?(.*)$", line)
            if numeric_prefix and "%" in numeric_prefix.group(1):
                repaired.append(line)
                continue
            if numeric_prefix and arabic_ratio(numeric_prefix.group(2)) >= 0.65:
                phone_or_number, body = numeric_prefix.groups()
                tokens = [_move_rtl_punctuation(token) for token in body.split()]
                restored = f"{' '.join(reversed(tokens))} - {phone_or_number}"
            else:
                tokens = [_move_rtl_punctuation(token) for token in line.split()]
                restored = " ".join(reversed(tokens))
            restored = re.sub(r"(?<!\w)-(\d+)\b", r"\1-", restored)
            repaired.append(restored)
        else:
            repaired.append(line)
    return "\n".join(repaired)


def _repair_table_cell(cell: str) -> str:
    """Repair one extracted RTL table cell without reversing numeric digits."""
    from rag.text_utils import arabic_ratio

    percent = re.match(r"^(\d[\d,./]*)%(.*)$", cell)
    if percent:
        number, suffix = percent.groups()
        inside = re.search(r"\((.*?)\)", suffix)
        if inside:
            words = " ".join(reversed(inside.group(1).split()))
            return f"{number}% ({words})"
        return f"{number}%{suffix}"

    trailing_percent = re.match(r"^(\d[\d,./]*)\s+(.+?)%$", cell)
    if trailing_percent:
        number, qualifier = trailing_percent.groups()
        qualifier = " ".join(reversed(qualifier.split()))
        return f"{qualifier} {number}%"

    if detect_reversed(cell, min_words=2).is_reversed:
        return reverse_lines(cell)
    if arabic_ratio(cell) >= 0.65:
        return " ".join(reversed([_move_rtl_punctuation(w) for w in cell.split()]))
    # Numeric lists in these RTL tables are emitted backwards as a sequence;
    # keep each number intact and restore only the item order.
    return " ".join(reversed([_move_rtl_punctuation(w) for w in cell.split()]))


def _move_rtl_punctuation(token: str) -> str:
    """Move punctuation that extraction placed on the visual-left side."""
    if token.startswith(")") and token.endswith("("):
        return "(" + token[1:-1] + ")"
    leading = ""
    while token and token[0] in ":،؛!?؟.!" :
        leading += token[0]
        token = token[1:]
    if leading:
        token += leading
    if token.startswith(")"):
        token = token[1:] + ")"
    if token.endswith("("):
        token = "(" + token[:-1]
    return token
