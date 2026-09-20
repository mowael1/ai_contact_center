"""Streaming loader for the Parquet knowledge base.

Reads either a single ``.parquet`` file or every ``.parquet`` under a
directory (default ``rag/data/``), in row-group batches so a 2 800-row /
24 MB corpus with 2.5 MB HTML blobs never has to be materialised at once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional, Sequence

import pyarrow.parquet as pq

from rag.config import settings
from rag.logging_utils import get_logger

logger = get_logger(__name__)

ALL_COLUMNS = [
    "source_url", "scrape_status", "raw_html", "raw_text", "llm_model",
    "llm_enhanced_text", "prompt_version", "scraped_at", "llm_processed_at",
    "processing_time_sec",
]

# Skipping raw_html cuts >90% of the bytes read when it is not needed.
LIGHT_COLUMNS = [
    "source_url", "scrape_status", "raw_text", "llm_enhanced_text",
    "scraped_at", "llm_model", "prompt_version",
]


def resolve_sources(
    file: Optional[Path] = None, directory: Optional[Path] = None
) -> list[Path]:
    """Resolve a --file / --directory pair into a sorted list of parquet paths."""
    if file:
        path = Path(file)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")
        return [path]
    base = Path(directory) if directory else settings.DATA_DIR
    if not base.exists():
        raise FileNotFoundError(f"Data directory not found: {base}")
    paths = sorted(base.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No .parquet files found under {base}")
    return paths


def count_rows(paths: Sequence[Path]) -> int:
    return sum(pq.ParquetFile(p).metadata.num_rows for p in paths)


def iter_records(
    paths: Sequence[Path],
    columns: Optional[Sequence[str]] = None,
    batch_size: int = 64,
    limit: Optional[int] = None,
) -> Iterator[dict]:
    """Yield raw parquet rows as dicts, annotated with their source file."""
    emitted = 0
    for path in paths:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        cols = [c for c in (columns or ALL_COLUMNS) if c in available]
        logger.debug("Reading %s (%d rows)", path.name, parquet.metadata.num_rows)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=cols):
            for row in batch.to_pylist():
                row["dataset_file"] = path.name
                yield row
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
