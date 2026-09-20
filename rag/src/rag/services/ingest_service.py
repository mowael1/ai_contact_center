"""End-to-end ingestion: Parquet -> sections -> chunks -> embeddings -> Chroma.

Idempotency contract
--------------------
Chunk ids are deterministic (see
:func:`rag.chunking.semantic_chunker.build_chunk_id`). Ingestion is therefore
run as *delete-then-upsert per document*: every chunk currently stored for a
document whose id is not in the freshly computed set is removed first, then the
new set is upserted. Re-running on unchanged input is a no-op in content terms
and can never create duplicate vectors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

from rag.chunking.semantic_chunker import SemanticChunker
from rag.config import settings
from rag.embeddings.base import EmbeddingService
from rag.htmlx.section_extractor import SectionExtractor
from rag.ingestion.document_builder import build_document
from rag.ingestion.parquet_loader import iter_records, resolve_sources
from rag.logging_utils import get_logger
from rag.models import Chunk, Document, Section
from rag.vectorstore.base import VectorStore

logger = get_logger(__name__)


def _chunking_embeddings(default: EmbeddingService) -> EmbeddingService:
    """Embedding backend for sentence-boundary detection.

    Boundary detection only needs *relative* similarity between neighbouring
    sentences, and its vectors are discarded. Since sentences outnumber stored
    chunks ~3:1, pointing this at a cheap local provider removes most of the
    hosted-API traffic. Defaults to the same service used for storage.
    """
    override = settings.CHUNKING_EMBEDDING_PROVIDER.strip()
    if not override:
        return default
    from rag.embeddings.factory import build_embedding_service

    service = build_embedding_service(provider=override)
    logger.info("Chunk-boundary embeddings use a separate provider: %s", service.model)
    return service


@dataclass
class IngestStats:
    rows_read: int = 0
    documents_usable: int = 0
    documents_skipped: int = 0
    sections: int = 0
    chunks_built: int = 0
    chunks_upserted: int = 0
    chunks_deleted: int = 0
    documents_unchanged: int = 0
    errors: list[str] = field(default_factory=list)
    elapsed_sec: float = 0.0
    extraction_methods: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rows_read": self.rows_read,
            "documents_usable": self.documents_usable,
            "documents_skipped": self.documents_skipped,
            "documents_unchanged": self.documents_unchanged,
            "sections": self.sections,
            "chunks_built": self.chunks_built,
            "chunks_upserted": self.chunks_upserted,
            "chunks_deleted": self.chunks_deleted,
            "extraction_methods": self.extraction_methods,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "errors": self.errors[:20],
            "error_count": len(self.errors),
        }


def iter_documents(
    file: Optional[Path] = None,
    directory: Optional[Path] = None,
    limit: Optional[int] = None,
    columns: Optional[Sequence[str]] = None,
    batch_size: int = 32,
) -> Iterator[Document]:
    """Stream usable, normalised documents from the Parquet corpus."""
    paths = resolve_sources(file, directory)
    for row in iter_records(paths, columns=columns, batch_size=batch_size, limit=limit):
        document = build_document(row)
        if document is not None:
            yield document


class IngestionPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        chunker: Optional[SemanticChunker] = None,
        extractor: Optional[SectionExtractor] = None,
    ) -> None:
        self.store = vector_store
        self.embeddings = embedding_service
        self.chunker = chunker or SemanticChunker(_chunking_embeddings(embedding_service))
        self.extractor = extractor or SectionExtractor()

    # -- per-document stages ------------------------------------------------
    def sections_for(self, document: Document) -> list[Section]:
        return self.extractor.extract(document.raw_html, document.retrieval_text)

    def chunks_for(self, document: Document) -> tuple[list[Section], list[Chunk]]:
        sections = self.sections_for(document)
        chunks = self.chunker.chunk_document(document, sections)
        return sections, chunks

    # -- full run -----------------------------------------------------------
    def run(
        self,
        file: Optional[Path] = None,
        directory: Optional[Path] = None,
        limit: Optional[int] = None,
        dry_run: bool = False,
        progress: Optional[callable] = None,
    ) -> IngestStats:
        stats = IngestStats()
        started = time.perf_counter()
        paths = resolve_sources(file, directory)
        logger.info("Ingesting from %d parquet file(s)", len(paths))

        # Stale-chunk pruning costs one vector-store round-trip per document.
        # On a first ingest into an empty collection there is nothing to prune,
        # so skip it entirely - it roughly halves the request count.
        self._skip_pruning = False
        if not dry_run:
            try:
                self._skip_pruning = self.store.count() == 0
            except Exception:
                self._skip_pruning = False
            if self._skip_pruning:
                logger.info("Collection is empty - skipping stale-chunk pruning")

        pending_chunks: list[Chunk] = []
        # Documents are staged so their sentences can be embedded in one batch.
        stage: list[tuple[Document, list[Section]]] = []
        stage_size = max(8, settings.INGEST_BATCH_SIZE // 8)

        def drain_stage() -> None:
            """Section-extract, chunk and queue every staged document."""
            nonlocal pending_chunks, stage
            if not stage:
                return
            try:
                chunk_lists = self.chunker.chunk_documents(stage)
            except Exception as exc:
                stats.errors.append(f"batch of {len(stage)}: {type(exc).__name__}: {exc}")
                logger.exception("Failed to chunk a batch of %d documents", len(stage))
                stage = []
                return
            for (document, sections), chunks in zip(stage, chunk_lists):
                stats.sections += len(sections)
                for section in sections:
                    key = section.extraction_method
                    stats.extraction_methods[key] = stats.extraction_methods.get(key, 0) + 1
                stats.chunks_built += len(chunks)
                if not dry_run and chunks:
                    stats.chunks_deleted += self._prune_stale(document, chunks)
                    pending_chunks.extend(chunks)
            stage = []

        for row in iter_records(paths, batch_size=16, limit=limit):
            stats.rows_read += 1
            document = build_document(row)
            if document is None:
                stats.documents_skipped += 1
                continue
            stats.documents_usable += 1
            try:
                sections = self.sections_for(document)
            except Exception as exc:
                stats.errors.append(f"{document.source_url}: {type(exc).__name__}: {exc}")
                logger.exception("Failed to extract sections from %s", document.source_url)
                continue

            stage.append((document, sections))
            if len(stage) >= stage_size:
                drain_stage()

            if not dry_run and len(pending_chunks) >= settings.INGEST_BATCH_SIZE:
                stats.chunks_upserted += self._flush(pending_chunks)
                pending_chunks = []
            if progress:
                progress(stats)

        drain_stage()

        if pending_chunks and not dry_run:
            stats.chunks_upserted += self._flush(pending_chunks)

        stats.elapsed_sec = time.perf_counter() - started
        logger.info("Ingestion complete: %s", stats.to_dict())
        return stats

    # -- helpers ------------------------------------------------------------
    def _prune_stale(self, document: Document, chunks: Sequence[Chunk]) -> int:
        """Delete chunks stored for this document that the new run did not produce."""
        if getattr(self, "_skip_pruning", False):
            return 0
        try:
            existing = set(self.store.get_document_chunk_ids(document.document_id))
        except NotImplementedError:
            return 0
        except Exception as exc:
            logger.warning("Could not list existing chunks for %s: %s", document.document_id, exc)
            return 0
        stale = existing - {c.chunk_id for c in chunks}
        if not stale:
            return 0
        try:
            removed = self.store.delete_chunks(sorted(stale))
        except Exception as exc:
            logger.warning("Stale-chunk pruning failed for %s: %s", document.document_id, exc)
            return 0
        logger.debug("Pruned %d stale chunks for %s", removed, document.document_id)
        return removed

    def _flush(self, chunks: list[Chunk]) -> int:
        vectors = self.embeddings.embed_documents([c.text for c in chunks])
        return self.store.upsert_chunks(chunks, vectors)
