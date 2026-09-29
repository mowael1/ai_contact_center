"""Chroma Cloud vector store.

Cloud only - there is deliberately no ``PersistentClient`` path and nothing is
ever written to disk. Credentials come exclusively from the environment and
are never logged.
"""

from __future__ import annotations

import time
from typing import Any, Optional, Sequence

from rag.config import settings
from rag.logging_utils import get_logger
from rag.models import Chunk, RetrievedChunk
from rag.vectorstore.base import VectorStore
from rag.vectorstore.keyword import lexical_score

logger = get_logger(__name__)

# Chroma rejects very large single requests; this keeps each round-trip small
# enough while still amortising network cost across many chunks.
DEFAULT_BATCH = 100
MAX_ATTEMPTS = 4
# Chroma Cloud's default Get quota rejects a request above 300 records.
MAX_LEXICAL_CANDIDATES = 300
MAX_DOCUMENT_PREDICATES = 8


class ChromaCloudStore(VectorStore):
    def __init__(
        self,
        collection_name: Optional[str] = None,
        client: Any = None,
        embedding_model: str = "",
    ) -> None:
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.embedding_model = embedding_model
        self._client = client if client is not None else self._build_client()
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            # Cosine matches the normalised embeddings used throughout.
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Chroma Cloud collection ready: %s", self.collection_name)

    # -- connection ---------------------------------------------------------
    @staticmethod
    def _build_client():
        if not settings.chroma_is_configured():
            raise RuntimeError(
                "Chroma Cloud is not configured. Set CHROMA_API_KEY, CHROMA_TENANT "
                "and CHROMA_DATABASE in rag/.env (see rag/.env.example)."
            )
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed. pip install chromadb") from exc

        # chromadb >=0.5.17 exposes CloudClient; older builds need HttpClient
        # against api.trychroma.com. Both are supported here so the project
        # works with whichever version is pinned.
        if hasattr(chromadb, "CloudClient"):
            return chromadb.CloudClient(
                api_key=settings.CHROMA_API_KEY,
                tenant=settings.CHROMA_TENANT,
                database=settings.CHROMA_DATABASE,
            )
        from chromadb.config import Settings as ChromaSettings

        return chromadb.HttpClient(
            host="api.trychroma.com",
            port=8000,
            ssl=True,
            tenant=settings.CHROMA_TENANT,
            database=settings.CHROMA_DATABASE,
            headers={"x-chroma-token": settings.CHROMA_API_KEY},
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    # -- writes -------------------------------------------------------------
    def add_chunks(self, chunks, embeddings) -> int:
        return self._write(chunks, embeddings, upsert=False)

    def upsert_chunks(self, chunks, embeddings) -> int:
        return self._write(chunks, embeddings, upsert=True)

    def _write(
        self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]], upsert: bool
    ) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}"
            )
        written = 0
        for start in range(0, len(chunks), DEFAULT_BATCH):
            batch = chunks[start : start + DEFAULT_BATCH]
            vectors = [list(v) for v in embeddings[start : start + DEFAULT_BATCH]]
            payload = {
                "ids": [c.chunk_id for c in batch],
                "documents": [c.text for c in batch],
                "metadatas": [c.to_metadata() for c in batch],
                "embeddings": vectors,
            }
            method = self._collection.upsert if upsert else self._collection.add
            self._retry(method, **payload)
            written += len(batch)
            logger.debug("Wrote %d/%d chunks", written, len(chunks))
        logger.info("%s %d chunks into %s", "Upserted" if upsert else "Added", written, self.collection_name)
        return written

    def delete_document(self, document_id: str) -> int:
        existing = self.get_document_chunk_ids(document_id)
        if not existing:
            return 0
        self.delete_chunks(existing)
        logger.info("Deleted %d chunks for document %s", len(existing), document_id)
        return len(existing)

    def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        ids = list(chunk_ids)
        for start in range(0, len(ids), DEFAULT_BATCH):
            self._retry(self._collection.delete, ids=ids[start : start + DEFAULT_BATCH])
        return len(ids)

    def list_documents(self) -> list[dict[str, Any]]:
        result = self._retry(self._collection.get, include=["metadatas"])
        by_doc: dict[str, dict[str, Any]] = {}
        for meta in (result.get("metadatas") or []):
            meta = meta or {}
            doc_id = meta.get("document_id")
            if not doc_id:
                continue
            entry = by_doc.setdefault(doc_id, {
                "document_id": doc_id,
                "source": meta.get("source", ""),
                "document_title": meta.get("document_title", ""),
                "source_type": meta.get("source_type", ""),
                "company_id": meta.get("company_id"),
                "chunks": 0,
            })
            entry["chunks"] += 1
        return sorted(by_doc.values(), key=lambda d: d["source"])

    def get_document_chunk_ids(self, document_id: str) -> list[str]:
        result = self._retry(
            self._collection.get,
            where={"document_id": document_id},
            include=[],
        )
        return list(result.get("ids") or [])

    # -- reads --------------------------------------------------------------
    def similarity_search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        where = build_where(filters)
        result = self._retry(
            self._collection.query,
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        return parse_query_result(result)

    def keyword_search(
        self,
        query_terms: Sequence[str],
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        # Chroma's document filter supplies an independent lexical candidate
        # set; ranking below uses normalized token overlap so Arabic hamza and
        # diacritic variants compare consistently.
        terms = list(dict.fromkeys(
            term for term in query_terms if len(term) >= 3 or term.isdigit()
        ))[:20]
        if not terms:
            return []
        search_terms = list(terms)
        for term in terms:
            if term.startswith("ا") and len(term) > 2:
                search_terms.extend(("أ" + term[1:], "إ" + term[1:], "آ" + term[1:]))
        search_terms = list(dict.fromkeys(search_terms))[:MAX_DOCUMENT_PREDICATES]
        clauses = [{"$contains": term} for term in search_terms]
        where_document = clauses[0] if len(clauses) == 1 else {"$or": clauses}
        result = self._retry(
            self._collection.get,
            where=build_where(filters) or None,
            where_document=where_document,
            limit=min(max(top_k * 10, 100), MAX_LEXICAL_CANDIDATES),
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        ranked = []
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            score = lexical_score(terms, text or "")
            if score <= 0:
                continue
            section_path = metadata.get("section_path") or ""
            ranked.append((score, RetrievedChunk(
                chunk_id=chunk_id,
                text=text or "",
                score=score,
                distance=1.0 - score,
                source_url=metadata.get("source_url", ""),
                section_title=metadata.get("section_title") or None,
                section_path=[part for part in section_path.split(" > ") if part] or None,
                document_id=metadata.get("document_id", ""),
                metadata=dict(metadata),
            )))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [chunk for _, chunk in ranked[:top_k]]

    def count(self) -> int:
        return int(self._retry(self._collection.count))

    def embedding_dimension(self) -> Optional[int]:
        """Return the collection's immutable vector width without downloading it.

        Chroma fixes a collection's dimension on its first insert.  Looking at
        one vector lets the composition root select the matching embedding
        backend for collections created before the configured model changed.
        """
        if self.count() == 0:
            return None
        result = self._retry(self._collection.peek, limit=1)
        vectors = result.get("embeddings")
        if vectors is None or len(vectors) == 0:
            return None
        vector = vectors[0]
        return len(vector) if vector is not None else None

    def info(self) -> dict[str, Any]:
        """Diagnostics with the API key deliberately excluded."""
        data: dict[str, Any] = {
            "backend": "chroma_cloud",
            "collection_name": self.collection_name,
            "tenant": settings.CHROMA_TENANT or "<unset>",
            "database": settings.CHROMA_DATABASE or "<unset>",
            "api_key": "<set>" if settings.CHROMA_API_KEY else "<unset>",
            "embedding_model": self.embedding_model or settings.EMBEDDING_MODEL,
        }
        try:
            data["vector_count"] = self.count()
            data["connected"] = True
        except Exception as exc:
            data["connected"] = False
            data["error"] = type(exc).__name__
        return data

    # -- transient failure handling ----------------------------------------
    @staticmethod
    def _retry(fn, *args, **kwargs):
        delay = 1.0
        last: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last = exc
                name = type(exc).__name__
                if not _is_transient(exc) or attempt == MAX_ATTEMPTS - 1:
                    raise
                logger.warning(
                    "Chroma call failed (%s), retry %d/%d", name, attempt + 1, MAX_ATTEMPTS
                )
                time.sleep(delay)
                delay *= 2
        raise last  # pragma: no cover


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(
        marker in text
        for marker in ("timeout", "connection", "temporarily", "503", "502", "429", "rate limit")
    )


def build_where(filters: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Translate simple equality/IN filters into Chroma's ``where`` syntax."""
    if not filters:
        return {}
    clauses: list[dict[str, Any]] = []
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values = [v for v in value if v is not None]
            if values:
                clauses.append({key: {"$in": list(values)}})
        else:
            clauses.append({key: {"$eq": value}})
    if not clauses:
        return {}
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def parse_query_result(result: dict[str, Any]) -> list[RetrievedChunk]:
    """Flatten Chroma's nested query response into RetrievedChunk objects."""
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    out: list[RetrievedChunk] = []
    for i, chunk_id in enumerate(ids):
        meta = dict(metadatas[i] or {}) if i < len(metadatas) else {}
        distance = float(distances[i]) if i < len(distances) else 0.0
        path_raw = meta.get("section_path") or ""
        out.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=documents[i] if i < len(documents) else "",
                # Cosine distance -> similarity score in [0, 1].
                score=round(max(0.0, 1.0 - distance), 6),
                distance=round(distance, 6),
                source_url=meta.get("source_url", ""),
                section_title=meta.get("section_title") or None,
                section_path=[p for p in path_raw.split(" > ") if p] or None,
                document_id=meta.get("document_id", ""),
                metadata=meta,
            )
        )
    return out
