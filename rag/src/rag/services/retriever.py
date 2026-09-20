"""Retrieval service - usable entirely independently of generation."""

from __future__ import annotations

import time
from typing import Any, Optional

from rag.config import settings
from rag.text_utils import content_hash
from rag.embeddings.base import EmbeddingService
from rag.logging_utils import get_logger
from rag.models import RetrievedChunk
from rag.vectorstore.base import VectorStore

logger = get_logger(__name__)

#: Metadata fields that may be used as retrieval filters. Kept to a short,
#: explicit list rather than allowing arbitrary keys.
ALLOWED_FILTERS = ("domain", "language", "document_id", "source_url", "dataset_file")


class RetrievalService:
    def __init__(self, vector_store: VectorStore, embedding_service: EmbeddingService):
        self.store = vector_store
        self.embeddings = embedding_service
        self.last_latency: dict[str, float] = {}

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        deduplicate: Optional[bool] = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.TOP_K
        clean_filters = sanitize_filters(filters)
        dedupe = settings.DEDUPLICATE_RESULTS if deduplicate is None else deduplicate

        # This corpus publishes the same copy under several URLs (e.g. the
        # "/en/vodafone-cash" and "/en/Vodafone-cash" variants), so ask for
        # extra candidates and collapse them after scoring.
        fetch_k = min(top_k * 3, 100) if dedupe else top_k

        t0 = time.perf_counter()
        vector = self.embeddings.embed_query(query)
        t1 = time.perf_counter()
        results = self.store.similarity_search(vector, top_k=fetch_k, filters=clean_filters)
        t2 = time.perf_counter()

        if dedupe:
            results = deduplicate_by_content(results)
        results = results[:top_k]

        self.last_latency = {
            "embed_ms": round((t1 - t0) * 1000, 3),
            "search_ms": round((t2 - t1) * 1000, 3),
            "total_ms": round((t2 - t0) * 1000, 3),
        }
        logger.info(
            "Retrieved %d chunks for query (%d chars) in %.1f ms",
            len(results), len(query), self.last_latency["total_ms"],
        )
        return results


def deduplicate_by_content(results: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Collapse results whose text is identical, keeping the best-scoring one.

    The duplicate's URL is preserved in ``metadata["duplicate_source_urls"]`` so
    an agent can still see every page the passage appears on.
    """
    best: dict[str, RetrievedChunk] = {}
    order: list[str] = []
    for result in results:
        key = result.metadata.get("content_hash") or content_hash(result.text.strip())
        existing = best.get(key)
        if existing is None:
            best[key] = result
            order.append(key)
            continue
        # Keep the higher-scoring copy; remember the other URL.
        keep, drop = (existing, result) if existing.score >= result.score else (result, existing)
        alternates = list(keep.metadata.get("duplicate_source_urls") or [])
        if drop.source_url and drop.source_url not in alternates:
            alternates.append(drop.source_url)
        keep.metadata = {**keep.metadata, "duplicate_source_urls": alternates}
        best[key] = keep
    return [best[k] for k in order]


def sanitize_filters(filters: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Drop unknown keys so a caller cannot craft an arbitrary Chroma filter."""
    if not filters:
        return {}
    out = {k: v for k, v in filters.items() if k in ALLOWED_FILTERS and v is not None}
    dropped = set(filters) - set(out)
    if dropped:
        logger.warning("Ignoring unsupported retrieval filters: %s", sorted(dropped))
    return out
