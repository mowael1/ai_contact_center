"""Vector store interface.

Nothing outside :mod:`rag.vectorstore` imports ``chromadb``. Swapping the
backend, or mocking it in tests, is therefore a single substitution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from rag.models import Chunk, RetrievedChunk


class VectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        """Insert chunks. Existing ids are overwritten."""

    @abstractmethod
    def upsert_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        """Insert or update chunks by their deterministic id."""

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """Remove every chunk belonging to a document. Returns count removed."""

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        """Nearest-neighbour search with optional metadata filtering."""

    @abstractmethod
    def count(self) -> int:
        """Number of vectors currently stored."""

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Connection / collection diagnostics. Must never include secrets."""

    @abstractmethod
    def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        """Remove specific chunks by id (used for stale-chunk pruning)."""

    def list_documents(self) -> list[dict[str, Any]]:
        """Distinct documents indexed in this collection."""
        raise NotImplementedError

    def get_document_chunk_ids(self, document_id: str) -> list[str]:
        """Ids currently stored for a document (used for stale-chunk pruning)."""
        raise NotImplementedError
