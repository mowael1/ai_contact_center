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

    @property
    def path_string(self) -> str:
        if not self.section_path:
            return "(no section)"
        return " > ".join(self.section_path)


@dataclass(slots=True)
class Chunk:
    """The unit that is embedded, stored in Chroma Cloud and retrieved."""

    chunk_id: str
    document_id: str
    source_url: str
    text: str
    section_title: Optional[str]
    section_path: Optional[list[str]]
    heading_level: Optional[int]
    section_index: int
    chunk_index: int
    chunking_method: ChunkingMethod
    section_extraction_method: SectionExtractionMethod
    language: str
    domain: str
    scraped_at: Optional[str] = None
    dataset_file: Optional[str] = None
    component_type: Optional[str] = None
    sentence_count: int = 0
    content_hash: str = ""

    def char_len(self) -> int:
        return len(self.text)

    def to_metadata(self) -> dict[str, Any]:
        """Compact metadata for Chroma.

        Chroma only accepts scalar metadata values, so ``section_path`` is
        serialised as a ``>``-joined string. The raw HTML is never included.
        """
        meta: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source_url": self.source_url,
            "section_title": self.section_title or "",
            "section_path": " > ".join(self.section_path) if self.section_path else "",
            "heading_level": self.heading_level if self.heading_level is not None else -1,
            "section_index": self.section_index,
            "chunk_index": self.chunk_index,
            "chunking_method": self.chunking_method,
            "section_extraction_method": self.section_extraction_method,
            "language": self.language,
            "domain": self.domain,
            "scraped_at": self.scraped_at or "",
            "dataset_file": self.dataset_file or "",
            "component_type": self.component_type or "",
            "sentence_count": self.sentence_count,
            "content_hash": self.content_hash,
        }
        return meta

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
    source_url: str
    section_title: Optional[str]
    section_path: Optional[list[str]]
    chunk_id: str
    document_id: str


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
