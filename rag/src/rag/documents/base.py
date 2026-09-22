"""Loaded-document representation, shared by every file-format loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class PageText:
    page_number: int          # 1-based, as a reader would cite it
    text: str


@dataclass(slots=True)
class HeadingCandidate:
    text: str
    page_number: int
    level: int
    order: int                # position in reading order
    reason: str               # font_size | bold | numbering | markdown | toc


@dataclass(slots=True)
class LoadedDocument:
    """A file turned into text, with enough structure to chunk it."""

    title: str
    source: str               # the filename, as it appears in chunk metadata
    path: str                 # absolute path, for traceability
    source_type: str          # pdf | docx | txt | md
    pages: list[PageText] = field(default_factory=list)
    headings: list[HeadingCandidate] = field(default_factory=list)
    byte_size: int = 0
    #: Populated when extraction is suspect - see quality.py
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip()).strip()

    def summary(self) -> dict:
        return {
            "title": self.title,
            "source": self.source,
            "source_type": self.source_type,
            "pages": self.page_count,
            "chars": len(self.full_text),
            "headings": len(self.headings),
            "warnings": self.warnings,
        }


def infer_title(path: Path, first_text: str = "") -> str:
    """Prefer a plausible first line, else a humanised filename."""
    for line in (first_text or "").splitlines():
        line = line.strip()
        if 3 <= len(line) <= 120 and not line.endswith(('.', '،')):
            return line
    return path.stem.replace("-", " ").replace("_", " ").strip()
