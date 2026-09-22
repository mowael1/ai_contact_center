"""The linear RAG pipeline: retrieve -> build context -> generate -> cite.

This is intentionally a plain function-style orchestrator, not a graph. See
README, "Why not LangGraph (yet)".
"""

from __future__ import annotations

from typing import Any, Optional

from rag.config import settings
from rag.logging_utils import get_logger
from rag.models import RagAnswer
from rag.services.generator import GenerationService
from rag.services.retriever import RetrievalService

logger = get_logger(__name__)


class RagService:
    def __init__(self, retriever: RetrievalService, generator: GenerationService):
        self.retriever = retriever
        self.generator = generator

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> RagAnswer:
        chunks = self.retriever.retrieve(question, top_k=top_k or settings.TOP_K, filters=filters)
        answer = self.generator.generate(question, chunks)
        answer.latency_ms = {**self.retriever.last_latency, **answer.latency_ms}
        answer.latency_ms["total_ms"] = round(
            self.retriever.last_latency.get("total_ms", 0.0)
            + answer.latency_ms.get("generate_ms", 0.0),
            3,
        )
        return answer
