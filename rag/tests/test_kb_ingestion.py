"""Document loading, chunking strategies and per-company ingestion."""

import json

import pytest

from rag.chunking.strategies import (
    ChunkingSettings,
    chunk_text,
    fixed_size_parts,
    has_usable_sections,
    section_based_parts,
)
from rag.chunking.tokenizer import get_token_counter
from rag.documents.arabic_repair import detect_reversed, repair, reverse_lines
from rag.documents.base import HeadingCandidate, LoadedDocument, PageText
from rag.documents.quality import assess, non_text_ratio, repair_extracted
from rag.documents.sections import build_sections
from rag.embeddings.local_providers import HashEmbedding
from rag.models import Section
from rag.services.kb_service import KnowledgeBaseService, build_chunk_id, document_id_for
from rag.services.tenancy import Tenant
from rag.vectorstore.memory_store import InMemoryVectorStore

CONFIG = ChunkingSettings(size=150, overlap=20, unit="tokens", min_section_units=1)


def doc(pages, headings=(), title="Doc", source="f.pdf"):
    return LoadedDocument(
        title=title, source=source, path=f"/kb/{source}", source_type="pdf",
        pages=[PageText(page_number=i + 1, text=t) for i, t in enumerate(pages)],
        headings=list(headings),
    )


def heading(text, level=1, page=1, order=0, reason="font_size"):
    return HeadingCandidate(text=text, page_number=page, level=level,
                            order=order, reason=reason)


# ---- token counting -------------------------------------------------------
def test_token_counter_units():
    assert get_token_counter("chars").count("abcd") == 4
    assert get_token_counter("words").count("a b c") == 3
    assert get_token_counter("tokens").count("hello world") > 0


def test_arabic_costs_more_tokens_than_english():
    """Documents the measured asymmetry that makes 150 tokens size-unequal."""
    counter = get_token_counter("tokens")
    arabic = "سياسة الاستبدال والاسترجاع لدى الشركة تتيح للعميل استبدال المنتج"
    english = "The company exchange and return policy lets a customer swap a product"
    ar_density = len(arabic) / counter.count(arabic)
    en_density = len(english) / counter.count(english)
    assert en_density > ar_density * 2


# ---- fixed-size chunking --------------------------------------------------
def test_fixed_size_respects_size_and_overlap():
    config = ChunkingSettings(size=10, overlap=3, unit="words")
    text = " ".join(f"w{i}" for i in range(40))
    parts = fixed_size_parts(text, config)
    assert all(p.unit_count <= 10 for p in parts)
    assert all(p.strategy == "fixed_size" for p in parts)
    # step = size - overlap
    assert len(parts) == pytest.approx(len(range(0, 40, 7)), abs=1)


def test_fixed_size_overlap_shares_context():
    config = ChunkingSettings(size=6, overlap=3, unit="words")
    parts = fixed_size_parts(" ".join(f"w{i}" for i in range(12)), config)
    first, second = parts[0].text.split(), parts[1].text.split()
    assert set(first) & set(second), "adjacent chunks must share tokens"


def test_fixed_size_loses_no_text():
    config = ChunkingSettings(size=8, overlap=2, unit="words")
    words = [f"w{i}" for i in range(31)]
    parts = fixed_size_parts(" ".join(words), config)
    seen = set()
    for part in parts:
        seen.update(part.text.split())
    assert seen == set(words)


def test_fixed_size_handles_empty_and_short_text():
    config = ChunkingSettings(size=10, overlap=2, unit="words")
    assert fixed_size_parts("", config) == []
    assert len(fixed_size_parts("only three words", config)) == 1


def test_overlap_cannot_exceed_size():
    config = ChunkingSettings(size=5, overlap=99, unit="words")
    parts = fixed_size_parts(" ".join(f"w{i}" for i in range(20)), config)
    assert len(parts) > 1, "a degenerate overlap must not loop forever"


# ---- section-based chunking ----------------------------------------------
def _section(index, title, text, page=1):
    section = Section(section_index=index, section_title=title, section_path=[title],
                      heading_level=2, text=text, extraction_method="pdf_font")
    section.page_number = page
    return section


def test_one_chunk_per_section():
    sections = [_section(i, f"S{i}", f"Body of section {i}. " * 3) for i in range(4)]
    parts = section_based_parts(sections, CONFIG)
    assert len(parts) == 4
    assert [p.part for p in parts] == [1, 2, 3, 4]
    assert all(p.strategy == "section_based" for p in parts)
    assert [p.section_title for p in parts] == ["S0", "S1", "S2", "S3"]


def test_oversized_section_is_split_but_stays_inside_the_section():
    config = ChunkingSettings(size=10, overlap=2, unit="words", min_section_units=1)
    sections = [_section(0, "Big", " ".join(f"w{i}" for i in range(60)))]
    parts = section_based_parts(sections, config)
    assert len(parts) > 1
    assert all(p.section_title == "Big" for p in parts)
    assert all(p.strategy == "section_based_split" for p in parts)


def test_page_number_is_carried_to_chunks():
    sections = [_section(0, "A", "text on page seven", page=7)]
    assert section_based_parts(sections, CONFIG)[0].page_number == 7


# ---- strategy selection ---------------------------------------------------
def test_sectioned_document_uses_section_based():
    text = "Intro body. " * 5 + "Second body. " * 5
    sections = [_section(0, "One", "Intro body. " * 5), _section(1, "Two", "Second body. " * 5)]
    _, strategy = chunk_text(text, sections, CONFIG)
    assert strategy == "section_based"


def test_unsectioned_document_falls_back_to_fixed_size():
    text = "Plain running text with no headings at all. " * 20
    _, strategy = chunk_text(text, [], CONFIG)
    assert strategy == "fixed_size"


def test_a_single_stray_heading_does_not_trigger_section_based():
    text = "x " * 500
    sections = [_section(0, "Stray", "tiny")]
    assert not has_usable_sections(sections, CONFIG, text)


# ---- section building from headings --------------------------------------
def test_sections_track_heading_hierarchy_and_pages():
    pages = ["Chapter One\nbody of chapter one\nSub A\nbody of sub a"]
    headings = [heading("Chapter One", level=1, order=0),
                heading("Sub A", level=2, order=2)]
    sections = build_sections(doc(pages, headings))
    paths = [s.section_path for s in sections]
    assert ["Chapter One"] in paths
    assert ["Chapter One", "Sub A"] in paths


def test_content_before_the_first_heading_is_untitled():
    pages = ["preamble text here\nHeading\nbody"]
    sections = build_sections(doc(pages, [heading("Heading", order=1)]))
    assert sections[0].section_title is None
    assert sections[0].extraction_method == "no_section"


def test_document_with_no_headings_yields_one_untitled_section():
    sections = build_sections(doc(["just body text"]))
    assert len(sections) == 1
    assert sections[0].section_title is None


# ---- extraction quality --------------------------------------------------
def test_broken_glyph_extraction_is_blocked():
    report = assess(doc(["·" * 400]))
    assert report.ok is False
    assert report.needs_ocr is True
    assert any("unusable" in p or "filler" in p for p in report.problems)


def test_empty_pdf_is_flagged_as_needing_ocr():
    report = assess(doc([""]))
    assert report.needs_ocr is True


def test_clean_text_passes():
    body = "Refunds above 5000 EGP require supervisor approval. " * 8
    report = assess(doc([body], [heading("Refunds")]))
    assert report.ok is True
    assert report.needs_ocr is False


def test_non_text_ratio_discriminates():
    assert non_text_ratio("·" * 40) == 1.0
    assert non_text_ratio("سياسة الاستبدال، خلال 14 يوماً.") < 0.1
    assert non_text_ratio("Refunds above 5000 EGP (see §2).") < 0.1


# ---- reversed Arabic -----------------------------------------------------
REVERSED = "عاجرتسالاو لادبتسالا ةسايس\n.مالتسالا خيرات نم ًاموي رشع ةعبرأ لالخ جتنملا لادبتسا ليمعلل نكمي"
CORRECT = reverse_lines(REVERSED)


def test_reversed_arabic_is_detected():
    verdict = detect_reversed(REVERSED)
    assert verdict.is_reversed is True
    assert verdict.reversed_score > verdict.forward_score


def test_correct_arabic_is_not_flagged():
    assert detect_reversed(CORRECT).is_reversed is False


def test_repair_restores_readable_arabic():
    repaired, verdict = repair(REVERSED)
    assert verdict.is_reversed
    assert "سياسة" in repaired and "الاستبدال" in repaired


def test_repair_leaves_correct_text_untouched():
    repaired, verdict = repair(CORRECT)
    assert repaired == CORRECT and not verdict.is_reversed


def test_english_is_never_treated_as_reversed():
    english = "Refunds above 5000 EGP require supervisor approval via the portal. " * 3
    assert detect_reversed(english).is_reversed is False


def test_repair_extracted_reports_what_it_did():
    _, info = repair_extracted(REVERSED)
    assert info["reversal"]["is_reversed"] is True


# ---- ids and tenancy -----------------------------------------------------
def test_document_id_differs_per_company():
    from pathlib import Path

    path = Path("/kb/manual.pdf")
    a = document_id_for(Tenant(company_id=1), path, "hash")
    b = document_id_for(Tenant(company_id=2), path, "hash")
    assert a != b, "two tenants uploading the same file must not collide"


def test_chunk_id_includes_the_company():
    a = build_chunk_id(Tenant(company_id=1), "doc", 1, "text")
    b = build_chunk_id(Tenant(company_id=2), "doc", 1, "text")
    assert a != b
    assert a.startswith("1:") and b.startswith("2:")


def test_ids_are_deterministic():
    first = build_chunk_id(Tenant(company_id=3), "doc", 2, "same text")
    second = build_chunk_id(Tenant(company_id=3), "doc", 2, "same text")
    assert first == second


# ---- end-to-end ingestion ------------------------------------------------
@pytest.fixture
def kb(tmp_path):
    store = InMemoryVectorStore("kb_company_1")
    service = KnowledgeBaseService(Tenant(company_id=1), store, HashEmbedding(), CONFIG)
    return service, store, tmp_path


def _write_md(tmp_path, name="policy.md"):
    path = tmp_path / name
    path.write_text(
        "# Refund policy\n\nItems may be refunded within 14 days of delivery.\n\n"
        "# Escalation\n\nRefunds above 5000 EGP need supervisor approval.\n",
        encoding="utf-8",
    )
    return path


def test_ingest_markdown_uses_section_based(kb):
    service, store, tmp_path = kb
    result = service.ingest_file(_write_md(tmp_path))
    assert result.status == "ingested"
    assert result.strategy == "section_based"
    assert store.count() == result.chunks > 1


def test_every_chunk_carries_the_company(kb):
    service, store, tmp_path = kb
    service.ingest_file(_write_md(tmp_path))
    for record in store._docs.values():
        assert record["metadata"]["company_id"] == 1


def test_reingesting_replaces_rather_than_duplicates(kb):
    service, store, tmp_path = kb
    path = _write_md(tmp_path)
    service.ingest_file(path)
    first = store.count()
    service.ingest_file(path)
    assert store.count() == first


def test_delete_document_removes_only_that_document(kb):
    service, store, tmp_path = kb
    a = service.ingest_file(_write_md(tmp_path, "a.md"))
    service.ingest_file(_write_md(tmp_path, "b.md"))
    total = store.count()
    removed = service.delete_document(a.document_id)
    assert removed == a.chunks
    assert store.count() == total - a.chunks


def test_list_documents_reports_each_file(kb):
    service, store, tmp_path = kb
    service.ingest_file(_write_md(tmp_path, "a.md"))
    service.ingest_file(_write_md(tmp_path, "b.md"))
    sources = {d["source"] for d in store.list_documents()}
    assert sources == {"a.md", "b.md"}


def test_chunk_export_matches_the_agreed_json_shape(kb):
    service, _, tmp_path = kb
    chunks, _, _, _ = service.chunk_file(_write_md(tmp_path))
    exported = chunks[0].to_export()
    assert set(exported) == {"text", "metadata"}
    for key in ("source", "path", "part"):
        assert key in exported["metadata"]
    assert json.dumps(exported, ensure_ascii=False)


def test_unsupported_file_type_fails_cleanly(kb, tmp_path):
    service, _, _ = kb
    bad = tmp_path / "image.png"
    bad.write_bytes(b"\x89PNG")
    result = service.ingest_file(bad)
    assert result.status == "failed"
    assert "Unsupported" in result.reason
