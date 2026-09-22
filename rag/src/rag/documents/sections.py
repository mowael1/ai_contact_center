"""Turn heading candidates into ordered sections with page provenance.

Shares the heading-stack approach used by the HTML extractor, so
``section_path`` means the same thing regardless of source format.
"""

from __future__ import annotations

from typing import Optional

from rag.documents.base import LoadedDocument
from rag.logging_utils import get_logger
from rag.models import Section

logger = get_logger(__name__)

_REASON_TO_METHOD = {
    "font_size": "pdf_font",
    "bold": "pdf_bold",
    "numbering": "pdf_numbering",
    "markdown": "markdown",
    "docx_style": "docx_style",
    "toc": "pdf_font",
}


def build_sections(document: LoadedDocument, min_chars: int = 1) -> list[Section]:
    """Split the document text at its headings, preserving hierarchy.

    Content before the first heading becomes an untitled ``no_section`` region
    rather than being attached to a heading it does not belong to.
    """
    text = document.full_text
    if not text.strip():
        return []
    if not document.headings:
        return [
            Section(section_index=0, section_title=None, section_path=None,
                    heading_level=None, text=text, extraction_method="no_section",
                    page_number=document.pages[0].page_number if document.pages else None)
        ]

    # Locate each heading in the flat text, in reading order.
    located: list[tuple[int, object]] = []
    cursor = 0
    for heading in sorted(document.headings, key=lambda h: h.order):
        position = text.find(heading.text, cursor)
        if position == -1:
            position = text.find(heading.text)
        if position == -1:
            continue
        located.append((position, heading))
        cursor = position + len(heading.text)
    located.sort(key=lambda item: item[0])
    if not located:
        return [
            Section(section_index=0, section_title=None, section_path=None,
                    heading_level=None, text=text, extraction_method="no_section")
        ]

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []

    preamble = text[: located[0][0]].strip()
    if len(preamble) >= min_chars:
        sections.append(
            Section(section_index=0, section_title=None, section_path=None,
                    heading_level=None, text=preamble, extraction_method="no_section",
                    page_number=document.pages[0].page_number if document.pages else None)
        )

    for i, (position, heading) in enumerate(located):
        end = located[i + 1][0] if i + 1 < len(located) else len(text)
        body = text[position + len(heading.text) : end].strip()
        if len(body) < min_chars:
            continue

        level = max(1, heading.level or 1)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading.text))

        sections.append(
            Section(
                section_index=0,
                section_title=heading.text,
                section_path=[title for _, title in stack],
                heading_level=level,
                text=body,
                extraction_method=_REASON_TO_METHOD.get(heading.reason, "text_fallback"),
                page_number=heading.page_number,
                page_range=_page_range(document, position, end),
            )
        )

    for index, section in enumerate(sections):
        section.section_index = index
    return sections


def _page_range(document: LoadedDocument, start: int, end: int) -> Optional[str]:
    """Which page numbers a character span covers, as "11" or "11-12"."""
    if not document.pages:
        return None
    offset, touched = 0, []
    for page in document.pages:
        length = len(page.text) + 2   # the "\n\n" join
        if offset < end and (offset + length) > start:
            touched.append(page.page_number)
        offset += length
    if not touched:
        return None
    return str(touched[0]) if len(touched) == 1 else f"{touched[0]}-{touched[-1]}"
