"""Dataset and HTML-extraction diagnostic reports."""

from __future__ import annotations

import statistics
from collections import Counter
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

from rag.evaluation.retrieval_metrics import percentile
from rag.htmlx.section_extractor import SectionExtractor
from rag.ingestion.document_builder import _clean_str, build_document
from rag.ingestion.parquet_loader import count_rows, iter_records, resolve_sources
from rag.logging_utils import get_logger
from rag.text_utils import content_hash, detect_language

logger = get_logger(__name__)


def _length_stats(values: list[int]) -> dict:
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 1),
        "median": round(statistics.median(values), 1),
        "min": min(values),
        "max": max(values),
        "p25": percentile(values, 25),
        "p75": percentile(values, 75),
        "p95": percentile(values, 95),
    }


def dataset_report(
    file: Optional[Path] = None,
    directory: Optional[Path] = None,
    limit: Optional[int] = None,
    samples: int = 0,
) -> dict:
    """Coverage, status, duplicates, language mix and field-length statistics."""
    paths = resolve_sources(file, directory)
    statuses: Counter = Counter()
    domains: Counter = Counter()
    languages: Counter = Counter()
    text_sources: Counter = Counter()

    total = missing_html = missing_text = missing_enhanced = 0
    html_len: list[int] = []
    text_len: list[int] = []
    enhanced_len: list[int] = []
    urls: Counter = Counter()
    content_hashes: Counter = Counter()
    usable = 0
    sample_rows: list[dict] = []

    for row in iter_records(paths, batch_size=16, limit=limit):
        total += 1
        status = _clean_str(row.get("scrape_status")) or "unknown"
        statuses[status] += 1

        raw_html = _clean_str(row.get("raw_html"))
        raw_text = _clean_str(row.get("raw_text"))
        enhanced = _clean_str(row.get("llm_enhanced_text"))
        missing_html += raw_html is None
        missing_text += raw_text is None
        missing_enhanced += enhanced is None
        if raw_html:
            html_len.append(len(raw_html))
        if raw_text:
            text_len.append(len(raw_text))
        if enhanced:
            enhanced_len.append(len(enhanced))

        url = _clean_str(row.get("source_url")) or ""
        urls[url] += 1
        if url:
            domains[urlparse(url).netloc] += 1

        document = build_document(row)
        if document is not None:
            usable += 1
            languages[document.language] += 1
            text_sources[document.retrieval_text_source] += 1
            content_hashes[document.content_hash] += 1
            if samples and len(sample_rows) < samples:
                sample_rows.append(document.summary())

    duplicate_urls = {u: c for u, c in urls.items() if c > 1}
    duplicate_content = sum(c - 1 for c in content_hashes.values() if c > 1)

    # How often llm_enhanced_text is written in a different script than the
    # original page - a real and material property of this corpus.
    return {
        "source": {
            "files": [p.name for p in paths],
            "file_count": len(paths),
            "declared_total_rows": count_rows(paths),
        },
        "rows": {
            "read": total,
            "usable_documents": usable,
            "unusable": total - usable,
        },
        "scrape_status": dict(statuses.most_common()),
        "missing_fields": {
            "raw_html": missing_html,
            "raw_text": missing_text,
            "llm_enhanced_text": missing_enhanced,
        },
        "duplicates": {
            "duplicate_url_count": len(duplicate_urls),
            "duplicate_urls_sample": list(duplicate_urls)[:10],
            "duplicate_content_rows": duplicate_content,
        },
        "lengths": {
            "raw_html": _length_stats(html_len),
            "raw_text": _length_stats(text_len),
            "llm_enhanced_text": _length_stats(enhanced_len),
        },
        "languages": dict(languages.most_common()),
        "retrieval_text_source": dict(text_sources.most_common()),
        "domains": dict(domains.most_common(15)),
        "samples": sample_rows,
    }


def extraction_report(
    file: Optional[Path] = None,
    directory: Optional[Path] = None,
    limit: int = 200,
) -> dict:
    """How well HTML-first section extraction performs across the corpus."""
    paths = resolve_sources(file, directory)
    extractor = SectionExtractor()

    methods: Counter = Counter()
    levels: Counter = Counter()
    components: Counter = Counter()
    per_doc_sections: list[int] = []
    section_lengths: list[int] = []
    parsed = with_sections = fallback_docs = no_section_docs = failed = 0

    for row in iter_records(paths, batch_size=16, limit=limit):
        document = build_document(row)
        if document is None:
            continue
        try:
            sections = extractor.extract(document.raw_html, document.retrieval_text)
            parsed += 1
        except Exception:
            failed += 1
            continue

        per_doc_sections.append(len(sections))
        doc_methods = {s.extraction_method for s in sections}
        if any(m in ("html_structure", "html_component") for m in doc_methods):
            with_sections += 1
        if "text_fallback" in doc_methods:
            fallback_docs += 1
        if not sections or doc_methods == {"no_section"}:
            no_section_docs += 1

        for section in sections:
            methods[section.extraction_method] += 1
            levels[str(section.heading_level)] += 1
            section_lengths.append(len(section.text))
            if section.component_type:
                components[section.component_type] += 1

    return {
        "documents": {
            "parsed_successfully": parsed,
            "parse_failures": failed,
            "with_html_sections": with_sections,
            "requiring_text_fallback": fallback_docs,
            "no_reliable_section": no_section_docs,
            "avg_sections_per_document": (
                round(statistics.fmean(per_doc_sections), 2) if per_doc_sections else 0
            ),
            "median_sections_per_document": (
                round(statistics.median(per_doc_sections), 1) if per_doc_sections else 0
            ),
        },
        "section_extraction_methods": dict(methods.most_common()),
        "component_types": dict(components.most_common()),
        "heading_level_distribution": dict(sorted(levels.items())),
        "section_length": _length_stats(section_lengths),
    }
