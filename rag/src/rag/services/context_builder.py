"""Builds the grounded context block handed to the LLM.

Each chunk becomes a numbered source carrying its section path and URL, so the
model can cite ``[1]`` and the citation builder can resolve that number back to
real retrieved metadata. The model is never asked to produce a URL itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rag.config import settings
from rag.models import Citation, RetrievedChunk


@dataclass(slots=True)
class BuiltContext:
    text: str
    used: list[RetrievedChunk]
    citations: list[Citation]

    @property
    def is_empty(self) -> bool:
        return not self.used


def build_context(
    chunks: Sequence[RetrievedChunk], max_chars: int | None = None
) -> BuiltContext:
    budget = max_chars or settings.MAX_CONTEXT_CHARS
    blocks: list[str] = []
    used: list[RetrievedChunk] = []
    citations: list[Citation] = []
    total = 0

    for chunk in chunks:
        header_bits = []
        if chunk.section_path:
            header_bits.append(" > ".join(chunk.section_path))
        elif chunk.section_title:
            header_bits.append(chunk.section_title)
        header_bits.append(chunk.source_url)
        header = " | ".join(b for b in header_bits if b)

        index = len(used) + 1
        block = f"[{index}] {header}\n{chunk.text}"
        if total + len(block) > budget and used:
            break
        blocks.append(block)
        total += len(block)
        used.append(chunk)
        citations.append(
            Citation(
                index=index,
                source_url=chunk.source_url,
                section_title=chunk.section_title,
                section_path=chunk.section_path,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
            )
        )
    return BuiltContext("\n\n".join(blocks), used, citations)
