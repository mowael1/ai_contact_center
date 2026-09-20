"""Normalises raw parquet rows into :class:`~rag.models.Document`."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

from rag.config import settings
from rag.logging_utils import get_logger
from rag.models import Document
from rag.text_utils import content_hash, detect_language, same_script, short_hash

logger = get_logger(__name__)


# Soft-404s: pages the scraper recorded as "success" that actually render the
# site's not-found template. ~15% of successful rows in this corpus.
SOFT_404_PATTERNS = (
    "لا يوجد صفحة مسجلة بهذا العنوان",
    "الصفحة غير موجودة",
    "page not found",
    "this page is not registered",
)


def is_soft_404(text: Optional[str]) -> bool:
    if not text:
        return False
    lowered = text[:1500].lower()
    return any(p.lower() in lowered for p in SOFT_404_PATTERNS)


def _clean_str(value) -> Optional[str]:
    """Parquet nulls arrive as None or float('nan') depending on the reader."""
    if value is None:
        return None
    if isinstance(value, float):
        return None
    text = str(value)
    return text if text.strip() else None


def document_id_for(source_url: str) -> str:
    """Stable id derived from the URL - the only natural key in this dataset."""
    return short_hash(source_url, length=20)


def select_retrieval_text(
    enhanced: Optional[str], raw: Optional[str], policy: Optional[str] = None
) -> tuple[str, str]:
    """Choose the textual representation used for the text-fallback path.

    ``enhanced_first`` follows the specification literally. ``language_aware``
    exists because ~88% of Arabic pages in this corpus have an
    ``llm_enhanced_text`` that was silently translated into English; it keeps
    those pages in their original language.
    """
    policy = policy or settings.RETRIEVAL_TEXT_POLICY
    enhanced = (enhanced or "").strip()
    raw = (raw or "").strip()

    if policy == "raw_only":
        return (raw, "raw_text") if raw else (enhanced, "llm_enhanced_text")
    if policy == "language_aware" and enhanced and raw and not same_script(enhanced, raw):
        return raw, "raw_text"
    if enhanced:
        return enhanced, "llm_enhanced_text"
    return raw, "raw_text"


def is_usable(row: dict) -> bool:
    """A row is usable when the scrape succeeded and some content survived."""
    if _clean_str(row.get("scrape_status")) != settings.SUCCESS_STATUS:
        return False
    if not any(
        _clean_str(row.get(f)) for f in ("raw_html", "raw_text", "llm_enhanced_text")
    ):
        return False
    # A soft-404 is a successful fetch of a useless page - exclude it so the
    # not-found template does not become ~100 near-identical chunks.
    return not is_soft_404(_clean_str(row.get("raw_text")))


def build_document(row: dict) -> Optional[Document]:
    if not is_usable(row):
        return None
    source_url = _clean_str(row.get("source_url")) or ""
    if not source_url:
        return None

    raw_html = _clean_str(row.get("raw_html"))
    raw_text = _clean_str(row.get("raw_text"))
    enhanced = _clean_str(row.get("llm_enhanced_text"))
    retrieval_text, source_field = select_retrieval_text(enhanced, raw_text)

    scraped_at = row.get("scraped_at")
    if scraped_at is not None and not isinstance(scraped_at, datetime):
        try:
            scraped_at = datetime.fromisoformat(str(scraped_at))
        except (TypeError, ValueError):
            scraped_at = None

    # Language is judged on the original text, never on a translation.
    language = detect_language(raw_text or retrieval_text or "")

    return Document(
        document_id=document_id_for(source_url),
        source_url=source_url,
        domain=urlparse(source_url).netloc,
        raw_html=raw_html,
        raw_text=raw_text,
        llm_enhanced_text=enhanced,
        retrieval_text=retrieval_text,
        retrieval_text_source=source_field,
        language=language,
        scrape_status=_clean_str(row.get("scrape_status")) or "",
        scraped_at=scraped_at,
        llm_model=_clean_str(row.get("llm_model")),
        prompt_version=_clean_str(row.get("prompt_version")),
        dataset_file=row.get("dataset_file"),
        content_hash=content_hash(raw_html or "", raw_text or "", enhanced or ""),
    )
