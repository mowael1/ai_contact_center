"""Internal domain representations shared by every stage of the pipeline.

The pipeline is deliberately a chain of pure-ish transformations:

    RawRecord -> Document -> Section -> Chunk -> RetrievedChunk -> RagAnswer
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Literal, Optional

SectionExtractionMethod = Literal[
    "html_structure",
    "html_component",
    "pdf_font",
    "pdf_bold",
    "pdf_numbering",
    "markdown",
    "docx_style",
    "text_fallback",
    "no_section",
]

ChunkingMethod = Literal["semantic", "size_split", "whole_section"]


@dataclass(slots=True)
class Document:
    """A single scraped page normalised into the internal representation."""

    document_id: str
    source_url: str
    domain: str
    raw_html: Optional[str]
    raw_text: Optional[str]
    llm_enhanced_text: Optional[str]
    retrieval_text: str
    retrieval_text_source: str  # llm_enhanced_text | raw_text
    language: str  # ar | en | mixed | unknown
    scrape_status: str
    scraped_at: Optional[datetime]
    llm_model: Optional[str] = None
    prompt_version: Optional[str] = None
    dataset_file: Optional[str] = None
    content_hash: str = ""

    def summary(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "source_url": self.source_url,
            "domain": self.domain,
            "language": self.language,
            "html_chars": len(self.raw_html or ""),
            "raw_text_chars": len(self.raw_text or ""),
            "enhanced_chars": len(self.llm_enhanced_text or ""),
            "retrieval_text_source": self.retrieval_text_source,
            "dataset_file": self.dataset_file,
        }


@dataclass(slots=True)
class Section:
    """A contiguous, heading-scoped region of a document."""

    section_index: int
    section_title: Optional[str]
    section_path: Optional[list[str]]
    heading_level: Optional[int]
    text: str
    extraction_method: SectionExtractionMethod
    component_type: Optional[str] = None  # accordion | tab | faq | None
    anchor_id: Optional[str] = None
    #: 1-based page the section starts on (PDF only).
    page_number: Optional[int] = None
    #: "11-12" when a section spans pages.
    page_range: Optional[str] = None

    @property
    def path_string(self) -> str:
        if not self.section_path:
            return "(no section)"
        return " > ".join(self.section_path)


@dataclass(slots=True)
class Chunk:
    """The unit that is embedded, stored in Chroma Cloud and retrieved."""

    chunk_id: str
    #: Tenant. Every chunk belongs to exactly one company - see
    #: rag.services.kb_service for how isolation is enforced.
    company_id: int
    text: str
    #: Filename, as it appears in chunk metadata for traceability.
    source: str
    #: Absolute path of the originating file.
    path: str
    #: 1-based position of this chunk within its document.
    part: int
    document_id: str
    document_title: str = ""
    source_type: str = "pdf"           # pdf | docx | txt | md | web
    chunking_strategy: str = "section_based"   # section_based | fixed_size | ...
    section_title: Optional[str] = None
    section_path: Optional[list[str]] = None
    heading_level: Optional[int] = None
    section_index: Optional[int] = None
    chunk_index: int = 0
    section_extraction_method: SectionExtractionMethod = "no_section"
    page_number: Optional[int] = None
    page_range: Optional[str] = None
    language: str = "unknown"
    unit_count: int = 0
    content_hash: str = ""
    uploaded_by: Optional[int] = None
    ingested_at: Optional[str] = None
    #: Web-scrape only; unused for uploaded files.
    source_url: Optional[str] = None

    def char_len(self) -> int:
        return len(self.text)

    def to_metadata(self) -> dict[str, Any]:
        """Compact metadata for Chroma.

        Chroma accepts scalars only, so ``section_path`` is serialised as a
        ``>``-joined string. Raw file bytes are never included.
        """
        return {
            "chunk_id": self.chunk_id,
            "company_id": self.company_id,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "source": self.source,
            "path": self.path,
            "part": self.part,
            "source_type": self.source_type,
            "chunking_strategy": self.chunking_strategy,
            "section_title": self.section_title or "",
            "section_path": " > ".join(self.section_path) if self.section_path else "",
            "heading_level": self.heading_level if self.heading_level is not None else -1,
            "section_index": self.section_index if self.section_index is not None else -1,
            "chunk_index": self.chunk_index,
            "section_extraction_method": self.section_extraction_method,
            "page_number": self.page_number if self.page_number is not None else -1,
            "page_range": self.page_range or "",
            "language": self.language,
            "unit_count": self.unit_count,
            "content_hash": self.content_hash,
            "source_url": self.source_url or "",
        }

    def to_export(self) -> dict[str, Any]:
        """chunks.json shape: text plus source/path/part provenance."""
        return {
            "text": self.text,
            "metadata": {
                "source": self.source,
                "path": self.path,
                "part": self.part,
                "company_id": self.company_id,
                "document_title": self.document_title,
                "section_title": self.section_title,
                "section_path": self.section_path,
                "page_number": self.page_number,
                "chunking_strategy": self.chunking_strategy,
                "chunk_id": self.chunk_id,
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk returned by the retriever, with its similarity score."""

    chunk_id: str
    text: str
    score: float
    distance: float
    source_url: str
    section_title: Optional[str]
    section_path: Optional[list[str]]
    document_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Citation:
    """A citation built strictly from retrieved metadata - never from the LLM."""

    index: int
    chunk_id: str
    document_id: str
    #: Filename for an uploaded document, e.g. "noon-pickup-faqs.pdf".
    source: str = ""
    document_title: str = ""
    section_title: Optional[str] = None
    section_path: Optional[list[str]] = None
    page_number: Optional[int] = None
    #: Web-scrape only; empty for uploaded files.
    source_url: str = ""

    def label(self) -> str:
        """Human-readable reference, e.g. "دليل الإجراءات > التجديد، صفحة 12"."""
        parts = []
        if self.section_path:
            parts.append(" > ".join(self.section_path))
        elif self.section_title:
            parts.append(self.section_title)
        origin = self.document_title or self.source or self.source_url
        if origin:
            parts.append(origin)
        text = " — ".join(p for p in parts if p)
        if self.page_number and self.page_number > 0:
            text = f"{text}, page {self.page_number}" if text else f"page {self.page_number}"
        return text or self.chunk_id


@dataclass(slots=True)
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    estimated_cost_usd: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }


@dataclass(slots=True)
class RagAnswer:
    query: str
    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    has_sufficient_context: bool
    usage: Optional[LLMUsage] = None
    latency_ms: dict[str, float] = field(default_factory=dict)
    #: Populated by the agentic graph: one record per retrieve/evaluate pass.
    attempts: list[dict[str, Any]] = field(default_factory=list)
    #: Human-readable audit trail of the nodes the graph executed.
    trace: list[str] = field(default_factory=list)
