"""Pydantic request/response models for the RAG API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class RetrievalFilters(BaseModel):
    """Optional metadata filters. Unknown keys are ignored by the retriever."""

    domain: Optional[str] = Field(None, examples=["web.vodafone.com.eg"])
    language: Optional[str] = Field(None, examples=["ar", "en", "mixed"])
    document_id: Optional[str] = None
    source_url: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000,
                       examples=["ازاي أجدد باقة الإنترنت؟"])
    top_k: int = Field(5, ge=1, le=50)
    filters: Optional[RetrievalFilters] = None


class RetrievedChunkModel(BaseModel):
    chunk_id: str
    text: str
    score: float
    distance: float
    source_url: str
    section_title: Optional[str] = None
    section_path: Optional[list[str]] = None
    document_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    count: int
    results: list[RetrievedChunkModel]
    latency_ms: dict[str, float] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=50)
    filters: Optional[RetrievalFilters] = None
    include_chunks: bool = Field(
        False, description="Return the full retrieved passages alongside the answer."
    )
    agentic: Optional[bool] = Field(
        None,
        description=(
            "Run the LangGraph retrieve/evaluate/rewrite loop. Defaults to "
            "AGENTIC_ENABLED. Set false to force the linear pipeline."
        ),
    )
    max_attempts: Optional[int] = Field(
        None, ge=1, le=5,
        description="Retrieval attempts including the first (agentic mode only).",
    )


class CitationModel(BaseModel):
    index: int
    chunk_id: str
    document_id: str
    source: str = Field("", description="Filename of the cited document")
    document_title: str = ""
    section_title: Optional[str] = None
    section_path: Optional[list[str]] = None
    page_number: Optional[int] = None
    source_url: str = Field("", description="Web-scrape only; empty for uploads")
    label: str = Field("", description="Ready-to-display reference string")


class UsageModel(BaseModel):
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: Optional[float] = None


class AttemptModel(BaseModel):
    """One retrieve/evaluate pass of the agentic loop."""

    attempt: int
    query: str
    retrieved: int = 0
    top_score: float = 0.0
    is_relevant: bool = False
    confidence: float = 0.0
    reason: str = ""
    missing: str = ""
    latency_ms: float = 0.0


class QueryResponse(BaseModel):
    query: str
    answer: str
    has_sufficient_context: bool
    citations: list[CitationModel]
    chunks: Optional[list[RetrievedChunkModel]] = None
    usage: Optional[UsageModel] = None
    latency_ms: dict[str, float] = Field(default_factory=dict)
    attempts: list[AttemptModel] = Field(
        default_factory=list,
        description="Agentic mode only: the loop's per-attempt audit trail.",
    )
    trace: list[str] = Field(
        default_factory=list, description="Agentic mode only: nodes executed, in order."
    )


class HealthResponse(BaseModel):
    status: str
    service: str = "rag"
    vector_store: dict[str, Any] = Field(default_factory=dict)
    embedding: dict[str, Any] = Field(default_factory=dict)
