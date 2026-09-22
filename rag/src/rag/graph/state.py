"""State carried through the agentic retrieval graph.

Everything the loop needs to make its next decision lives here, plus a full
audit trail (`attempts`, `trace`) so a support engineer can see exactly why an
answer was or was not produced.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict

from rag.models import Citation, LLMUsage, RetrievedChunk


class AttemptRecord(TypedDict, total=False):
    """One pass through retrieve -> evaluate."""

    attempt: int
    query: str
    retrieved: int
    top_score: float
    is_relevant: bool
    confidence: float
    reason: str
    missing: str
    latency_ms: float


class RagState(TypedDict, total=False):
    # -- inputs ------------------------------------------------------------
    question: str
    top_k: int
    filters: Optional[dict[str, Any]]
    max_attempts: int

    # -- loop state --------------------------------------------------------
    query: str                 # the (possibly rewritten) query in play
    attempt: int
    chunks: list[RetrievedChunk]
    is_relevant: bool
    reason: str
    missing: str
    confidence: float

    # -- outputs -----------------------------------------------------------
    answer: str
    citations: list[Citation]
    has_sufficient_context: bool
    usage: Optional[LLMUsage]

    # -- audit trail -------------------------------------------------------
    # ``operator.add`` lets nodes append without clobbering earlier entries.
    attempts: Annotated[list[AttemptRecord], operator.add]
    trace: Annotated[list[str], operator.add]
    latency_ms: dict[str, float]
