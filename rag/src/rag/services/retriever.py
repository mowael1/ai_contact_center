"""Retrieval service - usable entirely independently of generation."""

from __future__ import annotations

import time
import re
import unicodedata
from typing import Any, Optional

from rag.config import settings
from rag.text_utils import content_hash
from rag.embeddings.base import EmbeddingService
from rag.logging_utils import get_logger
from rag.models import RetrievedChunk
from rag.services.tenancy import Tenant, require_tenant
from rag.vectorstore.base import VectorStore

logger = get_logger(__name__)

#: Metadata fields that may be used as retrieval filters. Kept to a short,
#: explicit list rather than allowing arbitrary keys.
#: Filters a caller may set. ``company_id`` is deliberately NOT here - it is
#: injected from the authenticated tenant and can never be chosen by a client.
ALLOWED_FILTERS = (
    "language", "document_id", "source", "source_type", "page_number",
)


class RetrievalService:
    """Tenant-scoped retrieval.

    ``tenant`` is required: the service refuses an unscoped search rather than
    silently querying everything. In production it comes from the JWT subject's
    company; the store it is given is already that company's collection.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        tenant: Optional[Tenant] = None,
    ):
        self.store = vector_store
        self.embeddings = embedding_service
        self.tenant = tenant
        self.last_latency: dict[str, float] = {}

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        deduplicate: Optional[bool] = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or settings.TOP_K
        tenant = require_tenant(self.tenant)
        clean_filters = sanitize_filters(filters)
        # Second, independent guard: even inside the company's own collection
        # the query is filtered by company_id.
        clean_filters["company_id"] = tenant.company_id
        dedupe = settings.DEDUPLICATE_RESULTS if deduplicate is None else deduplicate

        # Keep semantic and lexical retrieval independent until fusion. A wider
        # pool lets exact numbers surface even when the Arabic paraphrase has a
        # weak embedding match.
        fetch_k = min(max(top_k * 10, 30), 100)

        t0 = time.perf_counter()
        vector = self.embeddings.embed_query(query)
        t1 = time.perf_counter()
        semantic = self.store.similarity_search(vector, top_k=fetch_k, filters=clean_filters)
        lexical_terms = _lexical_terms(query)
        keyword = self.store.keyword_search(
            sorted(lexical_terms), top_k=fetch_k, filters=clean_filters
        )
        t2 = time.perf_counter()

        keyword_weight = 2.0 if any(term.isdigit() for term in lexical_terms) else 1.25
        results = reciprocal_rank_fusion(
            semantic, keyword, keyword_weight=keyword_weight
        )
        if dedupe:
            results = deduplicate_by_content(results)
        results = diversify_sections(results, top_k)
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


_ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u0640]")
_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_ARABIC_SYNONYMS = {
    "اقل": {"أقل", "الحد", "الأدنى", "ادنى", "اكثر", "أكثر", "minimum", "min"},
    "ليمت": {"حد", "الحد", "limit", "minimum"},
    "اوردر": {"طلب", "طلبية", "الطلبية", "الطلبات", "order", "orders"},
    "order": {"طلب", "طلبية", "الطلبية", "الطلبات", "اوردر"},
    "طلبية": {"الطلبيات", "طلبيات", "طلبات", "order", "orders"},
    "الطلبية": {"الطلبيات", "طلبيات", "طلبات", "order", "orders"},
}
_ARABIC_STOPWORDS = {
    "هل", "ينفع", "ممكن", "في", "على", "من", "ايه", "هو", "هي", "ب", "ال",
}


def _normalize_term(term: str) -> str:
    term = unicodedata.normalize("NFKC", term).lower()
    term = _ARABIC_DIACRITICS.sub("", term)
    return term.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}))


def _lexical_terms(text: str) -> set[str]:
    terms = {_normalize_term(token) for token in _WORD_RE.findall(text)}
    # Arabic clitics and definite articles commonly differ between colloquial
    # questions and formal PDF text ("الطلبات" vs "طلبات").
    expanded = set(terms)
    for term in terms:
        if term.startswith("ال") and len(term) > 3:
            expanded.add(term[2:])
        if term in _ARABIC_SYNONYMS:
            expanded.update(_normalize_term(alias) for alias in _ARABIC_SYNONYMS[term])
    return expanded - {_normalize_term(word) for word in _ARABIC_STOPWORDS}


def reciprocal_rank_fusion(
    semantic: list[RetrievedChunk],
    keyword: list[RetrievedChunk],
    k: int = 60,
    keyword_weight: float = 1.25,
) -> list[RetrievedChunk]:
    """Merge independent ranked lists without comparing incompatible scores."""
    by_id: dict[str, RetrievedChunk] = {}
    fused: dict[str, float] = {}
    for list_index, ranked_list in enumerate((semantic, keyword)):
        weight = 1.0 if list_index == 0 else keyword_weight
        for rank, chunk in enumerate(ranked_list, start=1):
            # Keep the semantic result's cosine score when a chunk appears in
            # both lists; keyword scores are on a different scale.
            by_id.setdefault(chunk.chunk_id, chunk)
            fused[chunk.chunk_id] = fused.get(chunk.chunk_id, 0.0) + weight / (k + rank)
    for chunk_id, chunk in by_id.items():
        chunk.metadata = {**chunk.metadata, "hybrid_score": fused[chunk_id]}
    return sorted(by_id.values(), key=lambda chunk: fused[chunk.chunk_id], reverse=True)


def diversify_sections(results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Choose distinct sections first so overlapping chunks do not crowd out facts."""
    selected: list[RetrievedChunk] = []
    deferred: list[RetrievedChunk] = []
    seen: set[tuple[str, str]] = set()
    for chunk in results:
        section = chunk.metadata.get("section_index")
        if section is None or str(section) in {"", "-1"}:
            deferred.append(chunk)
            continue
        key = (chunk.document_id, str(section))
        if key in seen:
            deferred.append(chunk)
        else:
            seen.add(key)
            selected.append(chunk)
    selected.extend(deferred)
    return selected[:top_k]


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
