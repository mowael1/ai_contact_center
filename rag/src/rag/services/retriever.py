"""Retrieval service - usable entirely independently of generation."""

from __future__ import annotations

import math
import threading
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
        # Candidates kept after hybrid retrieval; the re-ranker narrows them
        # back down to top_k. Equals top_k when re-ranking is disabled.
        pool_k = _candidate_pool(top_k)
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
        results = diversify_sections(results, pool_k)
        results = results[:pool_k]

        # Retriever -> Re-Ranker -> (caller's) context builder. ``None``
        # means nothing was re-ranked, so the original order is kept.
        reranked = rerank_chunks(query, results, top_k)
        t3 = time.perf_counter()
        results = (reranked if reranked is not None else results)[:top_k]

        self.last_latency = {
            "embed_ms": round((t1 - t0) * 1000, 3),
            "search_ms": round((t2 - t1) * 1000, 3),
            "total_ms": round((t3 - t0) * 1000, 3),
        }
        if reranked is not None:
            self.last_latency["rerank_ms"] = round((t3 - t2) * 1000, 3)
        logger.info(
            "Retrieved %d chunks for query (%d chars) in %.1f ms",
            len(results), len(query), self.last_latency["total_ms"],
        )
        return results


# ---- Re-ranker ------------------------------------------------------------
_CROSS_ENCODERS: dict[str, Any] = {}
_CROSS_ENCODER_LOCK = threading.Lock()


def _candidate_pool(top_k: int) -> int:
    """How many chunks the retriever keeps for the re-ranker to choose from."""
    if not settings.RERANK_ENABLED:
        return top_k
    return max(top_k, settings.TOP_K_RETRIEVAL)


def _get_cross_encoder(model_name: str) -> Any:
    """Load the cross-encoder once per process.

    A failed load (package missing, model unreachable) is remembered as
    ``None`` so it costs one warning, not a retry on every query.
    """
    with _CROSS_ENCODER_LOCK:
        if model_name in _CROSS_ENCODERS:
            return _CROSS_ENCODERS[model_name]
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(model_name, max_length=512)
        except Exception as exc:
            logger.warning(
                "Re-ranker %s unavailable (%s: %s); keeping the retriever's "
                "original ranking.",
                model_name, type(exc).__name__, exc,
            )
            model = None
        _CROSS_ENCODERS[model_name] = model
        return model


def rerank_chunks(
    query: str, chunks: list[RetrievedChunk], top_n: int
) -> Optional[list[RetrievedChunk]]:
    """Order the retriever's candidates by cross-encoder relevance to ``query``.

    Returns the SAME ``RetrievedChunk`` objects (nothing is rebuilt or dropped
    apart from the cut to ``top_n``), best first, with the cross-encoder score
    added under ``metadata["rerank_score"]``. ``score`` is left untouched.

    Returns ``None`` when nothing was re-ranked - disabled, fewer than two
    candidates, bad configuration, model unavailable or inference failure - so
    the caller simply keeps the retriever's own order.
    """
    if not settings.RERANK_ENABLED or len(chunks) < 2 or top_n < 1:
        return None
    cap = settings.TOP_N_RERANK
    keep = min(top_n, cap) if cap > 0 else top_n

    model = _get_cross_encoder(settings.RERANK_MODEL)
    if model is None:
        return None
    try:
        scores = [
            float(s)
            for s in model.predict(
                [(query, chunk.text or "") for chunk in chunks],
                show_progress_bar=False,
            )
        ]
        if len(scores) != len(chunks) or not all(math.isfinite(s) for s in scores):
            raise ValueError("re-ranker returned an unusable score list")
    except Exception as exc:
        logger.warning(
            "Re-ranking failed (%s: %s); keeping the retriever's original ranking.",
            type(exc).__name__, exc,
        )
        return None

    # Blank passages go last; ties keep the retriever's order (stable).
    order = sorted(
        range(len(chunks)),
        key=lambda i: (not (chunks[i].text or "").strip(), -scores[i], i),
    )[:keep]
    out: list[RetrievedChunk] = []
    for i in order:
        chunks[i].metadata = {**chunks[i].metadata, "rerank_score": scores[i]}
        out.append(chunks[i])
    logger.info("Re-ranked %d candidates -> kept %d", len(chunks), len(out))
    return out


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
