"""Extension points for a future iterative retrieval workflow.

The current pipeline is linear (retrieve -> build context -> generate ->
cite). These protocols exist so that adding an evaluate/rewrite/retry loop
later does not require rewriting the core services. See README, "Future
LangGraph extension point".

They are intentionally left **unimplemented**: shipping a fake
``ContextEvaluator`` that always returns "relevant" would be worse than
shipping none.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, runtime_checkable

from rag.models import RagAnswer, RetrievedChunk


@dataclass(slots=True)
class ContextVerdict:
    is_relevant: bool
    reason: str = ""
    confidence: float = 0.0
    #: What the judge said was absent - fed to the rewriter on a retry.
    missing: str = ""


@runtime_checkable
class Retriever(Protocol):
    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[dict[str, Any]] = None
    ) -> list[RetrievedChunk]: ...


@runtime_checkable
class ContextEvaluator(Protocol):
    """Judges whether retrieved context can answer the question.

    Implement this to drive the YES/NO branch of the future graph.
    """

    def evaluate(self, query: str, chunks: Sequence[RetrievedChunk]) -> ContextVerdict: ...


@runtime_checkable
class QueryRewriter(Protocol):
    """Produces an improved query after a failed retrieval attempt."""

    def rewrite(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        attempt: int = 1,
        original_question: Optional[str] = None,
        missing: str = "",
    ) -> str: ...


@runtime_checkable
class Generator(Protocol):
    def generate(
        self, query: str, chunks: Sequence[RetrievedChunk]
    ) -> RagAnswer: ...
