"""Chunking strategies.

Two strategies, selected per document by structure:

1. **Section-based** - each detected section becomes one chunk. Most documents
   in this corpus are short and organised into clear sections, and a section
   usually covers a single topic, so one-section-per-chunk preserves semantic
   meaning and improves retrieval.

2. **Fixed-size** - for documents with no reliable section structure. A sliding
   window of ``CHUNK_SIZE_TOKENS`` with ``CHUNK_OVERLAP_TOKENS`` of overlap, so
   nothing is lost and adjacent chunks share context.

A section that is itself larger than the size limit is split with strategy 2
inside the section, so section boundaries are still never crossed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from rag.chunking.tokenizer import get_token_counter
from rag.config import settings
from rag.logging_utils import get_logger
from rag.models import Section

logger = get_logger(__name__)


@dataclass(slots=True)
class ChunkingSettings:
    size: int = 150            # in counting units
    overlap: int = 20
    unit: str = "tokens"       # tokens | words | chars
    encoding: str = "cl100k_base"
    #: A section shorter than this is merged into its neighbour rather than
    #: indexed as a near-empty chunk.
    min_section_units: int = 20

    @classmethod
    def from_settings(cls, **overrides) -> "ChunkingSettings":
        base = cls(
            size=settings.CHUNK_SIZE_TOKENS,
            overlap=settings.CHUNK_OVERLAP_TOKENS,
            unit=settings.CHUNK_UNIT,
            encoding=settings.CHUNK_TOKEN_ENCODING,
            min_section_units=settings.CHUNK_MIN_UNITS,
        )
        for key, value in overrides.items():
            if value is not None and hasattr(base, key):
                setattr(base, key, value)
        return base

    def to_dict(self) -> dict:
        return {
            "size": self.size, "overlap": self.overlap, "unit": self.unit,
            "encoding": self.encoding if self.unit == "tokens" else None,
            "min_section_units": self.min_section_units,
        }


@dataclass(slots=True)
class TextPart:
    """One chunk of text plus where it came from."""

    text: str
    part: int
    strategy: str                       # section_based | fixed_size
    section_title: Optional[str] = None
    section_path: Optional[list[str]] = None
    heading_level: Optional[int] = None
    section_index: Optional[int] = None
    page_number: Optional[int] = None
    page_range: Optional[str] = None
    unit_count: int = 0


def fixed_size_parts(
    text: str,
    config: ChunkingSettings,
    start_part: int = 1,
    **carry,
) -> list[TextPart]:
    """Sliding window over ``text``. Overlap is applied between windows."""
    counter = get_token_counter(config.unit, config.encoding)
    units = counter.split(text)
    if not units:
        return []

    size = max(1, config.size)
    overlap = max(0, min(config.overlap, size - 1))
    step = size - overlap

    parts: list[TextPart] = []
    start = 0
    while start < len(units):
        window = units[start : start + size]
        body = counter.join(window).strip()
        if body:
            parts.append(
                TextPart(
                    text=body,
                    part=start_part + len(parts),
                    strategy="fixed_size",
                    unit_count=len(window),
                    **carry,
                )
            )
        if start + size >= len(units):
            break
        start += step
    return parts


def section_based_parts(
    sections: Sequence[Section],
    config: ChunkingSettings,
) -> list[TextPart]:
    """One chunk per section; oversized sections are windowed inside themselves."""
    counter = get_token_counter(config.unit, config.encoding)
    parts: list[TextPart] = []

    for section in sections:
        body = (section.text or "").strip()
        if not body:
            continue
        carry = {
            "section_title": section.section_title,
            "section_path": section.section_path,
            "heading_level": section.heading_level,
            "section_index": section.section_index,
            "page_number": getattr(section, "page_number", None),
            "page_range": getattr(section, "page_range", None),
        }
        total = counter.count(body)
        if total <= config.size:
            parts.append(
                TextPart(
                    text=body, part=len(parts) + 1, strategy="section_based",
                    unit_count=total, **carry,
                )
            )
            continue
        # Too big for one chunk: window it, but stay inside the section.
        windows = fixed_size_parts(body, config, start_part=len(parts) + 1, **carry)
        for w in windows:
            w.strategy = "section_based_split"
        parts.extend(windows)

    # Renumber so ``part`` is a clean 1..N sequence for the document.
    for i, part in enumerate(parts, start=1):
        part.part = i
    return parts


def has_usable_sections(
    sections: Sequence[Section], config: ChunkingSettings, full_text: str
) -> bool:
    """Decide whether a document is "organised into clear sections".

    Requires at least two titled sections that together cover most of the
    document, so a stray heading on an otherwise unstructured file does not
    trigger section-based chunking.
    """
    titled = [s for s in sections if s.section_title and (s.text or "").strip()]
    if len(titled) < 2:
        return False
    covered = sum(len(s.text) for s in titled)
    total = max(1, len(full_text.strip()))
    return covered / total >= 0.5


def chunk_text(
    full_text: str,
    sections: Sequence[Section],
    config: Optional[ChunkingSettings] = None,
) -> tuple[list[TextPart], str]:
    """Apply the appropriate strategy. Returns (parts, strategy_used)."""
    config = config or ChunkingSettings.from_settings()
    if sections and has_usable_sections(sections, config, full_text):
        parts = section_based_parts(sections, config)
        if parts:
            return parts, "section_based"
    parts = fixed_size_parts(full_text, config)
    return parts, "fixed_size"
