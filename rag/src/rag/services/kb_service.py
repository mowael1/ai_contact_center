"""Knowledge-base ingestion for uploaded files, scoped to one company.

Pipeline per file:

    load (pdf/docx/txt/md)
        -> quality gate
        -> headings -> sections (with page provenance)
        -> section_based OR fixed_size chunking
        -> embeddings
        -> Chroma collection kb_company_{id}
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from rag.chunking.strategies import ChunkingSettings, chunk_text
from rag.config import settings
from rag.documents.base import LoadedDocument
from rag.documents.loader import iter_supported_files, load_document
from rag.documents.quality import QualityReport, assess, repair_extracted
from rag.documents.sections import build_sections
from rag.embeddings.base import EmbeddingService
from rag.logging_utils import get_logger
from rag.models import Chunk
from rag.services.tenancy import Tenant, require_tenant
from rag.text_utils import content_hash, detect_language, short_hash
from rag.vectorstore.base import VectorStore

logger = get_logger(__name__)


def document_id_for(tenant: Tenant, file_path: Path, file_hash: str) -> str:
    """Stable per-company document id.

    ``company_id`` is part of the hash so two tenants uploading the identical
    file never collide - without it, deleting one tenant's copy would remove
    the other's.
    """
    return short_hash(str(tenant.company_id), file_path.name, file_hash, length=20)


def build_chunk_id(tenant: Tenant, document_id: str, part: int, text: str) -> str:
    return (
        f"{tenant.company_id}:{document_id}:{part}:"
        f"{short_hash(str(tenant.company_id), text, length=12)}"
    )


@dataclass
class FileResult:
    source: str
    path: str
    status: str                      # ingested | skipped | failed
    chunks: int = 0
    pages: int = 0
    strategy: str = ""
    sections: int = 0
    document_id: str = ""
    quality: Optional[dict] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, "")}


@dataclass
class IngestReport:
    company_id: int
    collection: str
    files_seen: int = 0
    files_ingested: int = 0
    files_failed: int = 0
    files_skipped: int = 0
    total_chunks: int = 0
    elapsed_sec: float = 0.0
    chunking: dict = field(default_factory=dict)
    results: list[FileResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "company_id": self.company_id,
            "collection": self.collection,
            "files_seen": self.files_seen,
            "files_ingested": self.files_ingested,
            "files_failed": self.files_failed,
            "files_skipped": self.files_skipped,
            "total_chunks": self.total_chunks,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "chunking": self.chunking,
            "results": [r.to_dict() for r in self.results],
        }


class KnowledgeBaseService:
    """Ingests and manages one company's uploaded documents."""

    def __init__(
        self,
        tenant: Tenant,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        chunking: Optional[ChunkingSettings] = None,
    ) -> None:
        self.tenant = require_tenant(tenant)
        self.store = vector_store
        self.embeddings = embedding_service
        self.chunking = chunking or ChunkingSettings.from_settings()

    # -- single file --------------------------------------------------------
    def chunk_file(self, path: str | Path) -> tuple[list[Chunk], LoadedDocument, QualityReport, str]:
        """Load, gate, section and chunk one file. No embedding, no writing."""
        path = Path(path)
        document = load_document(path)

        # Normalise presentation forms and undo reversed-Arabic extraction
        # before anything downstream sees the text.
        for page in document.pages:
            page.text, info = repair_extracted(page.text)
            if info["reversal"]["is_reversed"]:
                document.warnings.append(
                    f"page {page.page_number}: repaired reversed Arabic"
                )
        if document.warnings:
            # Headings and the title were derived from the raw layout, so they
            # need the same correction as the page text.
            for heading in document.headings:
                heading.text = heading.text[::-1]
            document.title = document.title[::-1]

        report = assess(document)
        if not report.ok:
            return [], document, report, "blocked"

        sections = build_sections(document, min_chars=self.chunking.min_section_units)
        parts, strategy = chunk_text(document.full_text, sections, self.chunking)

        file_hash = content_hash(document.full_text)
        document_id = document_id_for(self.tenant, path, file_hash)
        now = datetime.now(timezone.utc).isoformat()

        chunks: list[Chunk] = []
        for part in parts:
            chunks.append(
                Chunk(
                    chunk_id=build_chunk_id(self.tenant, document_id, part.part, part.text),
                    company_id=self.tenant.company_id,
                    text=part.text,
                    source=document.source,
                    path=document.path,
                    part=part.part,
                    document_id=document_id,
                    document_title=document.title,
                    source_type=document.source_type,
                    chunking_strategy=part.strategy,
                    section_title=part.section_title,
                    section_path=part.section_path,
                    heading_level=part.heading_level,
                    section_index=part.section_index,
                    chunk_index=part.part - 1,
                    section_extraction_method=_method_for(sections, part.section_index),
                    page_number=part.page_number,
                    page_range=part.page_range,
                    language=detect_language(part.text),
                    unit_count=part.unit_count,
                    content_hash=short_hash(part.text, length=32),
                    ingested_at=now,
                )
            )
        return chunks, document, report, strategy

    def ingest_file(self, path: str | Path, replace: bool = True) -> FileResult:
        path = Path(path)
        try:
            chunks, document, quality, strategy = self.chunk_file(path)
        except Exception as exc:
            logger.exception("Failed to process %s", path.name)
            return FileResult(source=path.name, path=str(path), status="failed",
                              reason=f"{type(exc).__name__}: {exc}")

        if not chunks:
            return FileResult(
                source=document.source, path=document.path, status="failed",
                pages=document.page_count, quality=quality.to_dict(),
                reason="; ".join(quality.problems) or "no chunks produced",
            )

        document_id = chunks[0].document_id
        if replace:
            removed = self.store.delete_document(document_id)
            if removed:
                logger.info("Replaced %d existing chunks for %s", removed, document.source)

        vectors = self.embeddings.embed_documents([c.text for c in chunks])
        self.store.upsert_chunks(chunks, vectors)
        logger.info(
            "Ingested %s: %d chunks (%s) into %s",
            document.source, len(chunks), strategy, self.tenant.collection_name,
        )
        return FileResult(
            source=document.source, path=document.path, status="ingested",
            chunks=len(chunks), pages=document.page_count, strategy=strategy,
            sections=len({c.section_index for c in chunks if c.section_index is not None}),
            document_id=document_id, quality=quality.to_dict(),
        )

    # -- batch --------------------------------------------------------------
    def ingest_path(self, target: str | Path, replace: bool = True) -> IngestReport:
        target = Path(target)
        files = [target] if target.is_file() else iter_supported_files(target)
        report = IngestReport(
            company_id=self.tenant.company_id,
            collection=self.tenant.collection_name,
            chunking=self.chunking.to_dict(),
            files_seen=len(files),
        )
        started = time.perf_counter()
        for path in files:
            result = self.ingest_file(path, replace=replace)
            report.results.append(result)
            if result.status == "ingested":
                report.files_ingested += 1
                report.total_chunks += result.chunks
            elif result.status == "skipped":
                report.files_skipped += 1
            else:
                report.files_failed += 1
        report.elapsed_sec = time.perf_counter() - started
        return report

    # -- management ---------------------------------------------------------
    def delete_document(self, document_id: str) -> int:
        return self.store.delete_document(document_id)

    def list_documents(self) -> list[dict]:
        """Distinct documents currently indexed for this company."""
        return self.store.list_documents()

    def count(self) -> int:
        return self.store.count()


def _method_for(sections: Sequence, section_index: Optional[int]) -> str:
    if section_index is None:
        return "no_section"
    for section in sections:
        if section.section_index == section_index:
            return section.extraction_method
    return "no_section"
