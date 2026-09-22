"""Shared object factories for the test suite."""

from __future__ import annotations


def make_test_chunk(
    part: int = 1,
    text: str = "some chunk text",
    company_id: int = 1,
    document_id: str = "doc1",
    source: str = "manual.pdf",
    section_title: str | None = "Renewal",
    section_path: list[str] | None = None,
    **overrides,
):
    """Build a valid Chunk for tests without repeating every field."""
    from rag.models import Chunk

    defaults = dict(
        chunk_id=f"{company_id}:{document_id}:{part}:h{part}",
        company_id=company_id,
        text=text,
        source=source,
        path=f"/kb/{company_id}/{source}",
        part=part,
        document_id=document_id,
        document_title="Manual",
        source_type="pdf",
        chunking_strategy="section_based",
        section_title=section_title,
        section_path=section_path if section_path is not None else ["Internet", "Renewal"],
        heading_level=2,
        section_index=0,
        chunk_index=part - 1,
        section_extraction_method="pdf_font",
        page_number=1,
        language="en",
    )
    defaults.update(overrides)
    return Chunk(**defaults)
