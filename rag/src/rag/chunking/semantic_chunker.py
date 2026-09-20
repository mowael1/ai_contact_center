"""Semantic chunking inside HTML sections.

Algorithm (no LLM anywhere):

1. A section is the hard outer boundary. Chunks are **never** merged across
   sections, however similar the adjoining sentences are.
2. Split the section into sentences (:mod:`rag.chunking.sentence_segmenter`).
3. Embed every sentence in one batch.
4. Compute cosine similarity between each adjacent pair.
5. A boundary is *proposed* where similarity drops - judged both against the
   absolute ``SEMANTIC_SIMILARITY_THRESHOLD`` and against the section's own
   similarity distribution, so a uniformly-similar section is not chopped up
   arbitrarily.
6. Walk the sentences accumulating a chunk, and **commit** a proposed boundary
   only when the chunk already satisfies ``CHUNK_MIN_SIZE``. Force a break at
   ``CHUNK_MAX_SIZE`` regardless of similarity; prefer to break at or after
   ``CHUNK_TARGET_SIZE``.
7. Re-attach a trailing chunk that ended up under ``CHUNK_MIN_SIZE`` to its
   predecessor, so tiny orphan chunks do not reach the index.
8. Apply ``CHUNK_OVERLAP`` (in sentences) between consecutive chunks of the
   same section to preserve cross-boundary context.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Optional, Sequence

from rag.chunking.sentence_segmenter import split_sentences
from rag.config import settings
from rag.embeddings.base import EmbeddingService
from rag.logging_utils import get_logger
from rag.models import Chunk, ChunkingMethod, Document, Section
from rag.text_utils import short_hash

logger = get_logger(__name__)


@dataclass(slots=True)
class ChunkingConfig:
    target_size: int = 0
    min_size: int = 0
    max_size: int = 0
    similarity_threshold: float = 0.0
    overlap_sentences: int = 0

    @classmethod
    def from_settings(cls, **overrides) -> "ChunkingConfig":
        base = cls(
            target_size=settings.CHUNK_TARGET_SIZE,
            min_size=settings.CHUNK_MIN_SIZE,
            max_size=settings.CHUNK_MAX_SIZE,
            similarity_threshold=settings.SEMANTIC_SIMILARITY_THRESHOLD,
            overlap_sentences=settings.CHUNK_OVERLAP,
        )
        for key, value in overrides.items():
            if value is not None and hasattr(base, key):
                setattr(base, key, value)
        return base

    def to_dict(self) -> dict:
        return {
            "target_size": self.target_size,
            "min_size": self.min_size,
            "max_size": self.max_size,
            "similarity_threshold": self.similarity_threshold,
            "overlap_sentences": self.overlap_sentences,
        }


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if not na or not nb:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


class SemanticChunker:
    """Turns sections into embedded-ready chunks."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        config: Optional[ChunkingConfig] = None,
    ) -> None:
        self.embeddings = embedding_service
        self.config = config or ChunkingConfig.from_settings()

    # -- public API ---------------------------------------------------------
    def chunk_document(self, document: Document, sections: Sequence[Section]) -> list[Chunk]:
        """Chunk every section of a document, embedding all sentences at once."""
        per_section: list[list[str]] = []
        for section in sections:
            per_section.append(split_sentences(section.text))

        flat = [s for group in per_section for s in group]
        vectors: list[list[float]] = []
        if flat:
            vectors = self.embeddings.embed_documents(flat)

        chunks: list[Chunk] = []
        cursor = 0
        for section, sentences in zip(sections, per_section):
            section_vectors = vectors[cursor : cursor + len(sentences)]
            cursor += len(sentences)
            chunks.extend(self._chunk_section(document, section, sentences, section_vectors))
        return _dedupe_within_document(chunks)

    # -- internals ----------------------------------------------------------
    def _chunk_section(
        self,
        document: Document,
        section: Section,
        sentences: list[str],
        vectors: list[list[float]],
    ) -> list[Chunk]:
        text = section.text.strip()
        if not text:
            return []

        cfg = self.config
        if not sentences:
            return []

        # A short section stays whole - splitting it would only create orphans.
        if len(text) <= cfg.max_size and len(sentences) == 1:
            return [self._build(document, section, text, 0, "whole_section", 1)]

        sentences, vectors = self._split_oversized(sentences, vectors)
        groups, method = self._group_sentences(sentences, vectors)
        groups = self._apply_overlap(groups, sentences)

        out: list[Chunk] = []
        for index, (start, end, overlap_start) in enumerate(groups):
            body = " ".join(sentences[overlap_start:end]).strip()
            if not body:
                continue
            out.append(
                self._build(document, section, body, index, method, end - overlap_start)
            )
        return out

    def _split_oversized(
        self, sentences: list[str], vectors: list[list[float]]
    ) -> tuple[list[str], list[list[float]]]:
        """Hard-wrap any single sentence longer than ``CHUNK_MAX_SIZE``.

        A handful of pages in this corpus carry a 3 000-character run-on
        "sentence" (terms-and-conditions blocks scraped without punctuation).
        Without this, such a sentence would silently breach the size contract.
        """
        limit = self.config.max_size
        if all(len(s) <= limit for s in sentences):
            return sentences, vectors

        out: list[str] = []
        reembed = False
        for sentence in sentences:
            if len(sentence) <= limit:
                out.append(sentence)
                continue
            reembed = True
            out.extend(_wrap_on_whitespace(sentence, limit))

        if not reembed:
            return out, vectors
        try:
            return out, self.embeddings.embed_documents(out)
        except Exception as exc:  # fall back to size-only splitting
            logger.warning("Re-embedding after hard-wrap failed: %s", type(exc).__name__)
            return out, []

    def _group_sentences(
        self, sentences: list[str], vectors: list[list[float]]
    ) -> tuple[list[tuple[int, int]], ChunkingMethod]:
        """Return [(start, end)) sentence ranges plus the method actually used."""
        cfg = self.config
        lengths = [len(s) for s in sentences]
        total = sum(lengths) + len(sentences) - 1

        if total <= cfg.max_size:
            return [(0, len(sentences))], "whole_section"

        have_vectors = len(vectors) == len(sentences) and all(vectors)
        method: ChunkingMethod = "semantic" if have_vectors else "size_split"

        similarities: list[float] = []
        if have_vectors:
            similarities = [
                cosine(vectors[i], vectors[i + 1]) for i in range(len(sentences) - 1)
            ]

        drop_at = self._boundary_flags(similarities) if have_vectors else []

        groups: list[tuple[int, int]] = []
        start = 0
        size = 0
        for i, sentence_len in enumerate(lengths):
            projected = size + sentence_len + (1 if size else 0)

            # Hard cap: never exceed max_size, unless a single sentence does.
            if size and projected > cfg.max_size:
                groups.append((start, i))
                start, size = i, sentence_len
                continue

            size = projected
            is_last = i == len(sentences) - 1
            if is_last:
                continue

            semantic_break = bool(drop_at) and drop_at[i]
            # Commit a semantic break only once the chunk can stand alone.
            if semantic_break and size >= cfg.min_size:
                groups.append((start, i + 1))
                start, size = i + 1, 0
                continue
            # Otherwise close on reaching the target size.
            if size >= cfg.target_size:
                groups.append((start, i + 1))
                start, size = i + 1, 0

        if start < len(sentences):
            groups.append((start, len(sentences)))

        groups = self._merge_orphans(groups, lengths, cfg.min_size, cfg.max_size)
        return groups, method

    @staticmethod
    def _boundary_flags(similarities: list[float]) -> list[bool]:
        """Mark positions where adjacent similarity drops meaningfully.

        Uses the absolute configured threshold OR a distribution-relative
        drop (one standard deviation below the section mean), so the algorithm
        adapts to sections that are uniformly dense or uniformly diverse.
        """
        if not similarities:
            return []
        cfg_threshold = settings.SEMANTIC_SIMILARITY_THRESHOLD
        if len(similarities) >= 3:
            mean = statistics.fmean(similarities)
            stdev = statistics.pstdev(similarities)
            relative = mean - stdev
        else:
            relative = -1.0
        return [(s < cfg_threshold) or (s < relative) for s in similarities]

    @staticmethod
    def _merge_orphans(
        groups: list[tuple[int, int]], lengths: list[int], min_size: int, max_size: int
    ) -> list[tuple[int, int]]:
        """Fold under-sized chunks into a neighbour when it does not overflow."""
        if len(groups) <= 1:
            return groups
        out = list(groups)
        i = 0
        while i < len(out):
            start, end = out[i]
            size = sum(lengths[start:end]) + max(0, end - start - 1)
            if size >= min_size or len(out) == 1:
                i += 1
                continue
            prev_size = (
                sum(lengths[out[i - 1][0] : out[i - 1][1]]) if i > 0 else None
            )
            if i > 0 and prev_size is not None and prev_size + size <= max_size:
                out[i - 1] = (out[i - 1][0], end)
                out.pop(i)
                continue
            if i + 1 < len(out):
                nxt_size = sum(lengths[out[i + 1][0] : out[i + 1][1]])
                if nxt_size + size <= max_size:
                    out[i + 1] = (start, out[i + 1][1])
                    out.pop(i)
                    continue
            i += 1
        return out

    def _apply_overlap(
        self, groups: list[tuple[int, int]], sentences: list[str]
    ) -> list[tuple[int, int, int]]:
        """Extend each chunk backwards by N sentences, bounded by max_size."""
        overlap = max(0, self.config.overlap_sentences)
        out: list[tuple[int, int, int]] = []
        for index, (start, end) in enumerate(groups):
            overlap_start = start
            if overlap and index > 0:
                candidate = max(groups[index - 1][0], start - overlap)
                span = sentences[candidate:end]
                size = sum(len(s) for s in span) + max(0, len(span) - 1)
                if size <= self.config.max_size:
                    overlap_start = candidate
            out.append((start, end, overlap_start))
        return out

    def _build(
        self,
        document: Document,
        section: Section,
        text: str,
        chunk_index: int,
        method: ChunkingMethod,
        sentence_count: int,
    ) -> Chunk:
        chunk_id = build_chunk_id(document, section, chunk_index, text)
        return Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            source_url=document.source_url,
            text=text,
            section_title=section.section_title,
            section_path=section.section_path,
            heading_level=section.heading_level,
            section_index=section.section_index,
            chunk_index=chunk_index,
            chunking_method=method,
            section_extraction_method=section.extraction_method,
            language=document.language,
            domain=document.domain,
            scraped_at=document.scraped_at.isoformat() if document.scraped_at else None,
            dataset_file=document.dataset_file,
            component_type=section.component_type,
            sentence_count=sentence_count,
            content_hash=short_hash(text, length=32),
        )


def _dedupe_within_document(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks whose text already appeared in the same document.

    Carousels, tab duplicates and repeated promo blocks otherwise emit the
    identical passage several times per page, which wastes vectors and lets one
    document crowd out the top-k.
    """
    seen: set[str] = set()
    out: list[Chunk] = []
    for chunk in chunks:
        key = chunk.content_hash or chunk.text.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out


def build_chunk_id(
    document: Document, section: Section, chunk_index: int, text: str
) -> str:
    """Stable, deterministic chunk id.

    Derived from document identity + section identity + chunk position +
    content hash. Re-running ingestion on unchanged input therefore produces
    byte-identical ids, which is what makes upserts idempotent. A content
    change yields a new id, so stale chunks are removed by the
    delete-then-upsert path in :class:`rag.services.ingest_service`.
    """
    section_key = section.section_path and " > ".join(section.section_path) or ""
    return (
        f"{document.document_id}:{section.section_index}:{chunk_index}:"
        f"{short_hash(section_key, text, length=12)}"
    )


def _wrap_on_whitespace(text: str, limit: int) -> list[str]:
    """Break a very long string at whitespace, never mid-word."""
    words = text.split()
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for word in words:
        extra = len(word) + (1 if current else 0)
        if current and size + extra > limit:
            parts.append(" ".join(current))
            current, size = [word], len(word)
        else:
            current.append(word)
            size += extra
    if current:
        parts.append(" ".join(current))
    return parts or [text[:limit]]
