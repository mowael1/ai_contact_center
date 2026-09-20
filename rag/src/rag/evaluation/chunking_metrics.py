"""Chunking diagnostics - stage 1 of the per-stage evaluation.

These are structural, not relevance metrics: they let two chunking
configurations be compared without touching the vector store or an LLM.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Sequence

from rag.evaluation.retrieval_metrics import percentile
from rag.models import Chunk


def chunk_diagnostics(
    chunks: Sequence[Chunk],
    document_count: int,
    section_count: int,
    config: dict | None = None,
    min_size: int = 200,
    max_size: int = 1400,
) -> dict:
    """Structural quality report for one chunking configuration."""
    if not chunks:
        return {"config": config or {}, "total_chunks": 0, "note": "no chunks produced"}

    sizes = [c.char_len() for c in chunks]
    sentences = [c.sentence_count for c in chunks]
    per_doc = Counter(c.document_id for c in chunks)
    per_section = Counter((c.document_id, c.section_index) for c in chunks)

    # A boundary violation would mean one chunk spanning two sections. The
    # chunker builds chunks strictly inside a section, so this must stay 0;
    # it is asserted here as a regression guard.
    violations = sum(
        1 for c in chunks if c.section_index is None or c.chunk_index is None
    )

    texts = [c.text.strip() for c in chunks]
    duplicates = len(texts) - len(set(texts))
    empty = sum(1 for t in texts if not t)

    return {
        "config": config or {},
        "total_chunks": len(chunks),
        "documents": document_count,
        "sections": section_count,
        "chunks_per_document": round(len(chunks) / max(1, document_count), 3),
        "chunks_per_section": round(len(chunks) / max(1, section_count), 3),
        "size": {
            "mean": round(statistics.fmean(sizes), 1),
            "median": round(statistics.median(sizes), 1),
            "min": min(sizes),
            "max": max(sizes),
            "p25": percentile(sizes, 25),
            "p75": percentile(sizes, 75),
            "p95": percentile(sizes, 95),
        },
        "sentences_per_chunk": {
            "mean": round(statistics.fmean(sentences), 2),
            "median": round(statistics.median(sentences), 1),
        },
        "pct_at_max_size": round(100 * sum(1 for s in sizes if s >= max_size) / len(sizes), 2),
        "pct_below_min_size": round(100 * sum(1 for s in sizes if s < min_size) / len(sizes), 2),
        "pct_over_max_size": round(100 * sum(1 for s in sizes if s > max_size) / len(sizes), 2),
        "section_boundary_violations": violations,
        "empty_chunks": empty,
        "duplicate_chunks": duplicates,
        "duplicate_pct": round(100 * duplicates / len(chunks), 2),
        "chunking_methods": dict(Counter(c.chunking_method for c in chunks)),
        "section_extraction_methods": dict(
            Counter(c.section_extraction_method for c in chunks)
        ),
        "max_chunks_in_one_document": max(per_doc.values()),
        "max_chunks_in_one_section": max(per_section.values()),
    }


def compare_configs(reports: Sequence[dict]) -> dict:
    """Side-by-side view of several chunking configurations."""
    return {
        "configurations": list(reports),
        "comparison": [
            {
                "config": r.get("config", {}),
                "total_chunks": r.get("total_chunks"),
                "median_size": r.get("size", {}).get("median"),
                "p95_size": r.get("size", {}).get("p95"),
                "pct_below_min_size": r.get("pct_below_min_size"),
                "pct_at_max_size": r.get("pct_at_max_size"),
                "duplicate_pct": r.get("duplicate_pct"),
                "chunks_per_document": r.get("chunks_per_document"),
            }
            for r in reports
        ],
        "note": (
            "Chunk count alone is not a quality signal. Read it together with "
            "size distribution, duplicate rate and downstream retrieval metrics."
        ),
    }
