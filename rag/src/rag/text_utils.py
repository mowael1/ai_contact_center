"""Language detection and text normalisation shared across the pipeline."""

from __future__ import annotations

import hashlib
import re
import unicodedata

ARABIC_RANGE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
LATIN_RANGE = re.compile(r"[A-Za-z]")

# Arabic tatweel and diacritics add noise without adding meaning.
_TATWEEL = "ـ"
_DIACRITICS = re.compile(r"[ً-ْٰۖ-ۭ]")
_WS = re.compile(r"[ \t ‏‎]+")
_MULTINEWLINE = re.compile(r"\n{3,}")


def arabic_ratio(text: str) -> float:
    """Share of alphabetic characters that are Arabic script."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    arabic = sum(1 for c in letters if ARABIC_RANGE.match(c))
    return arabic / len(letters)


def detect_language(text: str) -> str:
    """Coarse, deterministic language label: ``ar`` / ``en`` / ``mixed`` / ``unknown``."""
    if not text or not text.strip():
        return "unknown"
    ratio = arabic_ratio(text)
    has_latin = bool(LATIN_RANGE.search(text))
    if ratio >= 0.65:
        return "ar"
    if ratio <= 0.10:
        return "en" if has_latin else "unknown"
    return "mixed"


def same_script(a: str, b: str) -> bool:
    """True when two texts are written in broadly the same script family."""
    la, lb = detect_language(a), detect_language(b)
    if "unknown" in (la, lb):
        return False
    if la == lb:
        return True
    # "mixed" is compatible with either pure script.
    return "mixed" in (la, lb)


def normalize_text(text: str) -> str:
    """Unicode-normalise and collapse whitespace without destroying structure."""
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = out.replace(_TATWEEL, "")
    out = _DIACRITICS.sub("", out)
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = _WS.sub(" ", out)
    out = "\n".join(line.strip() for line in out.split("\n"))
    out = _MULTINEWLINE.sub("\n\n", out)
    return out.strip()


def content_hash(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update((part or "").encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


def short_hash(*parts: str, length: int = 16) -> str:
    return content_hash(*parts)[:length]
