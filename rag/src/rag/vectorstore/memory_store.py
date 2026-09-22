"""In-memory vector store used by tests and offline CLI runs.

Mirrors :class:`~rag.vectorstore.chroma_cloud.ChromaCloudStore` semantics
(deterministic ids, upsert, metadata filtering) so the same service code can
be exercised without Chroma Cloud credentials. Not a production backend: it is
an exact brute-force scan and holds everything in RAM.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional, Sequence

from rag.models import Chunk, RetrievedChunk
from rag.vectorstore.base import VectorStore
from rag.vectorstore.chroma_cloud import build_where


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 1.0
    return 1.0 - (dot / (na * nb))


def _matches(meta: dict[str, Any], where: dict[str, Any]) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_matches(meta, clause) for clause in where["$and"])
    for key, condition in where.items():
        value = meta.get(key)
        if isinstance(condition, dict):
            if "$eq" in condition and value != condition["$eq"]:
                return False
            if "$in" in condition and value not in condition["$in"]:
                return False
        elif value != condition:
            return False
    return True


class InMemoryVectorStore(VectorStore):
    def __init__(self, collection_name: str = "in_memory", embedding_model: str = ""):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._docs: dict[str, dict[str, Any]] = {}

    def add_chunks(self, chunks, embeddings) -> int:
        return self.upsert_chunks(chunks, embeddings)

    def upsert_chunks(self, chunks: Sequence[Chunk], embeddings) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunk/embedding count mismatch")
        for chunk, vector in zip(chunks, embeddings):
            self._docs[chunk.chunk_id] = {
                "text": chunk.text,
                "metadata": chunk.to_metadata(),
                "embedding": list(vector),
            }
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        ids = self.get_document_chunk_ids(document_id)
        for chunk_id in ids:
            self._docs.pop(chunk_id, None)
        return len(ids)

    def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        removed = 0
        for chunk_id in chunk_ids:
            if self._docs.pop(chunk_id, None) is not None:
                removed += 1
        return removed

    def list_documents(self) -> list[dict[str, Any]]:
        by_doc: dict[str, dict[str, Any]] = {}
        for rec in self._docs.values():
            meta = rec["metadata"]
            doc_id = meta.get("document_id")
            if not doc_id:
                continue
            entry = by_doc.setdefault(doc_id, {
                "document_id": doc_id, "source": meta.get("source", ""),
                "document_title": meta.get("document_title", ""),
                "source_type": meta.get("source_type", ""),
                "company_id": meta.get("company_id"), "chunks": 0,
            })
            entry["chunks"] += 1
        return sorted(by_doc.values(), key=lambda d: d["source"])

    def get_document_chunk_ids(self, document_id: str) -> list[str]:
        return [
            cid for cid, rec in self._docs.items()
            if rec["metadata"].get("document_id") == document_id
        ]

    def similarity_search(self, query_embedding, top_k=5, filters=None):
        where = build_where(filters)
        scored = []
        for chunk_id, rec in self._docs.items():
            if not _matches(rec["metadata"], where):
                continue
            scored.append((_cosine_distance(query_embedding, rec["embedding"]), chunk_id, rec))
        scored.sort(key=lambda x: x[0])
        out: list[RetrievedChunk] = []
        for distance, chunk_id, rec in scored[:top_k]:
            meta = rec["metadata"]
            path_raw = meta.get("section_path") or ""
            out.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    text=rec["text"],
                    score=round(max(0.0, 1.0 - distance), 6),
                    distance=round(distance, 6),
                    source_url=meta.get("source_url", ""),
                    section_title=meta.get("section_title") or None,
                    section_path=[p for p in path_raw.split(" > ") if p] or None,
                    document_id=meta.get("document_id", ""),
                    metadata=dict(meta),
                )
            )
        return out

    def count(self) -> int:
        return len(self._docs)

    def info(self) -> dict[str, Any]:
        return {
            "backend": "in_memory",
            "collection_name": self.collection_name,
            "vector_count": self.count(),
            "embedding_model": self.embedding_model,
            "connected": True,
        }

    # -- persistence for offline CLI sessions -------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._docs), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, **kw) -> "InMemoryVectorStore":
        store = cls(**kw)
        if Path(path).exists():
            store._docs = json.loads(Path(path).read_text(encoding="utf-8"))
        return store
