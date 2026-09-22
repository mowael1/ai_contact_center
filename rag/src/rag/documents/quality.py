"""Extraction-quality gate.

Arabic PDFs are the hardest case in this pipeline: many have no text layer at
all (scans), and some store text with reversed character order or detached
presentation-form glyphs. If extraction is damaged, every downstream stage
inherits the damage, so a document is checked *before* it is chunked and
indexed rather than after someone complains about bad answers.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from rag.documents.base import LoadedDocument
from rag.text_utils import arabic_ratio

# Arabic presentation forms. Correctly extracted text uses the base block
# (U+0600-06FF); heavy use of these means the shaping was baked into the text.
PRESENTATION_FORMS = re.compile(r"[ﭐ-﷿ﹰ-﻿]")
# Replacement char / private use = a broken font-to-Unicode mapping.
BROKEN_GLYPHS = re.compile(r"[�-]")
WORD = re.compile(r"[\w؀-ۿ]+")

#: Punctuation and symbols that legitimately appear in Arabic/English prose.
_OK_PUNCT = set(".,;:!?()[]{}-–—'\"/%&@#*+=<>|~`^$_\\" + "\u060C\u061B\u061F\u066A\u066B\u066C\u066D\u00AB\u00BB\u2018\u2019\u201C\u201D\u2026")


def non_text_ratio(text: str) -> float:
    """Share of visible characters that are neither alphanumeric nor ordinary
    punctuation. A high value means the font-to-Unicode mapping failed and the
    page came out as dots, boxes or filler glyphs."""
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return 0.0
    bad = sum(1 for c in visible if not c.isalnum() and c not in _OK_PUNCT)
    return bad / len(visible)


@dataclass(slots=True)
class QualityReport:
    ok: bool
    chars: int
    pages: int
    pages_with_text: int
    arabic_ratio: float
    presentation_form_ratio: float
    broken_glyph_ratio: float
    mean_word_length: float
    alpha_ratio: float
    non_text_ratio: float
    reversed_arabic: bool
    headings_found: int
    needs_ocr: bool
    problems: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "needs_ocr": self.needs_ocr,
            "chars": self.chars, "pages": self.pages,
            "pages_with_text": self.pages_with_text,
            "arabic_ratio": round(self.arabic_ratio, 3),
            "presentation_form_ratio": round(self.presentation_form_ratio, 4),
            "broken_glyph_ratio": round(self.broken_glyph_ratio, 4),
            "mean_word_length": round(self.mean_word_length, 2),
            "alpha_ratio": round(self.alpha_ratio, 3),
            "non_text_ratio": round(self.non_text_ratio, 3),
            "reversed_arabic": self.reversed_arabic,
            "headings_found": self.headings_found,
            "problems": self.problems,
        }


def assess(document: LoadedDocument, min_chars_per_page: int = 80) -> QualityReport:
    text = document.full_text
    chars = len(text)
    pages = max(1, document.page_count)
    with_text = sum(1 for p in document.pages if len(p.text.strip()) >= min_chars_per_page)

    letters = [c for c in text if c.isalpha()] or [" "]
    pres = len(PRESENTATION_FORMS.findall(text)) / len(letters)
    broken = len(BROKEN_GLYPHS.findall(text)) / max(1, chars)
    words = WORD.findall(text)
    mean_len = sum(len(w) for w in words) / max(1, len(words))
    visible = [c for c in text if not c.isspace()] or [" "]
    alpha = sum(1 for c in visible if c.isalnum()) / len(visible)
    non_text = non_text_ratio(text)

    problems: list[str] = []
    needs_ocr = False

    if chars < min_chars_per_page:
        problems.append("almost no extractable text - likely a scanned PDF")
        needs_ocr = True
    elif with_text / pages < 0.5:
        problems.append(
            f"only {with_text}/{pages} pages have text - partially scanned"
        )
        needs_ocr = True

    # The decisive check: if barely any character is a letter or digit, the
    # text layer is unusable regardless of what else looks fine. This catches
    # glyph-mapping failures that render as dots, boxes or filler symbols.
    if chars >= min_chars_per_page and alpha < 0.30:
        problems.append(
            f"only {alpha:.0%} of visible characters are letters or digits - "
            "the text layer is unusable (broken font encoding); OCR required"
        )
        needs_ocr = True
    if non_text > 0.30:
        problems.append(
            f"{non_text:.0%} of characters are outside normal text ranges - "
            "extraction produced filler glyphs"
        )
    if pres > 0.15:
        problems.append(
            f"{pres:.0%} of letters are Arabic presentation forms - text was "
            "extracted with shaping baked in; normalise with NFKC before indexing"
        )
    if broken > 0.02:
        problems.append(
            f"{broken:.1%} replacement/private-use glyphs - broken font encoding"
        )
    # Arabic words average ~4-5 chars; ~1 means every letter became its own token.
    if words and mean_len < 1.8 and arabic_ratio(text) > 0.3:
        problems.append(
            f"mean word length {mean_len:.1f} - letters appear detached, "
            "the text layer is probably unusable"
        )
    # Reversed Arabic passes every other check while being unusable, so it is
    # tested for explicitly.
    from rag.documents.arabic_repair import detect_reversed

    reversal = detect_reversed(text)
    if reversal.is_reversed:
        problems.append(
            "Arabic text is character-reversed (visual order) - repairable by "
            f"reversing each line (confidence {reversal.confidence:.0%})"
        )

    if not document.headings:
        problems.append("no headings detected - fixed-size chunking will be used")

    blocking = [
        p for p in problems
        if "no extractable text" in p or "unusable" in p or "filler glyphs" in p
    ]
    return QualityReport(
        ok=not blocking, chars=chars, pages=pages, pages_with_text=with_text,
        arabic_ratio=arabic_ratio(text), presentation_form_ratio=pres,
        broken_glyph_ratio=broken, mean_word_length=mean_len,
        alpha_ratio=alpha, non_text_ratio=non_text,
        reversed_arabic=reversal.is_reversed,
        headings_found=len(document.headings), needs_ocr=needs_ocr,
        problems=problems,
    )


def normalize_extracted(text: str) -> str:
    """NFKC folds Arabic presentation forms back to their base letters."""
    return unicodedata.normalize("NFKC", text)


def repair_extracted(text: str) -> tuple[str, dict]:
    """Normalise, then undo reversed-Arabic extraction if detected.

    Returns the corrected text and a report of what was done, so ingestion can
    record it against the document instead of silently changing content.
    """
    from rag.documents.arabic_repair import repair

    normalized = normalize_extracted(text)
    repaired, verdict = repair(normalized)
    return repaired, {
        "nfkc_applied": normalized != text,
        "reversal": verdict.to_dict(),
    }
