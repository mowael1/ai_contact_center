"""Sentence segmentation and semantic chunking."""

import pytest

from rag.chunking.semantic_chunker import ChunkingConfig, SemanticChunker, cosine
from rag.chunking.sentence_segmenter import split_sentences
from rag.models import Section


# ---- sentence segmentation ------------------------------------------------
def test_english_sentence_splitting():
    text = "You can renew your bundle. Then pay via Vodafone Cash. Enjoy 30% extra!"
    assert len(split_sentences(text)) == 3


def test_arabic_sentence_splitting():
    text = "ازاي أجدد باقة الإنترنت؟ تقدر تجددها من التطبيق. السعر 50 جنيه."
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert sentences[0].endswith("؟")


def test_mixed_language_sentence_splitting():
    text = "Flex 70 بيديك 100 دقيقة. You can renew via *880#. مرحبا!"
    assert len(split_sentences(text)) == 3


def test_ussd_code_does_not_split():
    sentences = split_sentences("Dial *880*1# to renew. Then confirm.")
    assert len(sentences) == 2
    assert "*880*1#" in sentences[0]


def test_url_does_not_split():
    sentences = split_sentences("See https://web.vodafone.com.eg/en/flex for details. Thanks.")
    assert len(sentences) == 2
    assert "https://web.vodafone.com.eg/en/flex" in sentences[0]


def test_decimal_numbers_do_not_split():
    sentences = split_sentences("The bundle gives 1.5 GB and costs 70.50 EGP. Enjoy.")
    assert len(sentences) == 2
    assert "1.5" in sentences[0] and "70.50" in sentences[0]


def test_abbreviations_do_not_split():
    sentences = split_sentences("Dr. Ahmed approved it, e.g. for Flex. Done.")
    assert len(sentences) == 2


def test_bullet_lines_stay_separate():
    text = "- Activate subscriptions\n- WATCH IT on Flex 70\n- Anghami on Flex 100"
    assert len(split_sentences(text)) == 3


def test_empty_text_yields_no_sentences():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


# ---- semantic chunking ----------------------------------------------------
def _section(text, index=0, title="T", path=None, method="html_structure"):
    return Section(
        section_index=index,
        section_title=title,
        section_path=path or [title],
        heading_level=2,
        text=text,
        extraction_method=method,
    )


def test_short_section_stays_one_chunk(document, lexical_embeddings):
    chunker = SemanticChunker(lexical_embeddings, ChunkingConfig(700, 200, 1400, 0.55, 1))
    section = _section("Renew your bundle from the app. It takes one minute.")
    chunks = chunker.chunk_document(document, [section])
    assert len(chunks) == 1
    assert chunks[0].chunking_method == "whole_section"


def test_long_section_is_split_within_limits(document, lexical_embeddings):
    config = ChunkingConfig(400, 120, 700, 0.55, 0)
    text = " ".join(
        f"Sentence number {i} describes bundle option {i} in detail for customers." 
        for i in range(60)
    )
    chunker = SemanticChunker(lexical_embeddings, config)
    chunks = chunker.chunk_document(document, [_section(text)])
    assert len(chunks) > 1
    assert all(c.char_len() <= config.max_size for c in chunks)


def test_section_boundaries_are_never_crossed(document, lexical_embeddings):
    """Two sections with near-identical text must not merge into one chunk."""
    a_text = "Renew your internet bundle from the Ana Vodafone application today. " * 3
    b_text = "Renew your internet bundle from the Ana Vodafone application now. " * 3
    sections = [
        _section(a_text, index=0, title="A", path=["A"]),
        _section(b_text, index=1, title="B", path=["B"]),
    ]
    chunker = SemanticChunker(lexical_embeddings, ChunkingConfig(700, 200, 1400, 0.55, 1))
    chunks = chunker.chunk_document(document, sections)

    assert {c.section_index for c in chunks} == {0, 1}
    for chunk in chunks:
        expected = "A" if chunk.section_index == 0 else "B"
        assert chunk.section_path == [expected]


def test_identical_text_is_deduplicated_within_a_document(document, lexical_embeddings):
    """Carousel clones and repeated promo blocks must not be indexed twice."""
    repeated = "Save articles, set preferences, and get tailored content for you."
    sections = [
        _section(repeated, index=0, title="A", path=["A"]),
        _section(repeated, index=1, title="B", path=["B"]),
    ]
    chunker = SemanticChunker(lexical_embeddings, ChunkingConfig.from_settings())
    chunks = chunker.chunk_document(document, sections)
    assert len(chunks) == 1


def test_min_size_prevents_orphan_chunks(document, lexical_embeddings):
    config = ChunkingConfig(200, 180, 600, 0.99, 0)  # threshold forces many breaks
    text = " ".join(f"Topic {i} is entirely unrelated to anything else here." for i in range(20))
    chunks = SemanticChunker(lexical_embeddings, config).chunk_document(
        document, [_section(text)]
    )
    # Only the final chunk may fall below min_size.
    undersized = [c for c in chunks[:-1] if c.char_len() < config.min_size]
    assert not undersized


def test_oversized_single_sentence_is_hard_wrapped(document, lexical_embeddings):
    config = ChunkingConfig(300, 100, 400, 0.55, 0)
    text = "word " * 400  # one 2000-char run-on with no terminator
    chunks = SemanticChunker(lexical_embeddings, config).chunk_document(
        document, [_section(text)]
    )
    assert all(c.char_len() <= config.max_size for c in chunks)


def test_overlap_adds_preceding_sentence(document, lexical_embeddings):
    config = ChunkingConfig(120, 60, 300, 0.55, 1)
    text = " ".join(f"Distinct topic {i} about unrelated subject {i}." for i in range(12))
    with_overlap = SemanticChunker(lexical_embeddings, config).chunk_document(
        document, [_section(text)]
    )
    no_overlap = SemanticChunker(
        lexical_embeddings, ChunkingConfig(120, 60, 300, 0.55, 0)
    ).chunk_document(document, [_section(text)])
    assert sum(c.char_len() for c in with_overlap) >= sum(c.char_len() for c in no_overlap)


def test_chunk_ids_are_deterministic(document, lexical_embeddings):
    chunker = SemanticChunker(lexical_embeddings, ChunkingConfig.from_settings())
    section = _section("Renew your bundle from the app. It takes about one minute to finish.")
    first = chunker.chunk_document(document, [section])
    second = chunker.chunk_document(document, [section])
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_changed_text_changes_the_chunk_id(document, lexical_embeddings):
    chunker = SemanticChunker(lexical_embeddings, ChunkingConfig.from_settings())
    a = chunker.chunk_document(document, [_section("Original content for this section here.")])
    b = chunker.chunk_document(document, [_section("Updated content for this section here.")])
    assert a[0].chunk_id != b[0].chunk_id


def test_chunk_metadata_is_complete(document, lexical_embeddings):
    chunker = SemanticChunker(lexical_embeddings, ChunkingConfig.from_settings())
    section = _section("Some content here for the metadata test.", path=["Internet", "Renewal"])
    meta = chunker.chunk_document(document, [section])[0].to_metadata()
    for key in (
        "chunk_id", "document_id", "source_url", "section_title", "section_path",
        "heading_level", "section_index", "chunk_index", "chunking_method",
        "section_extraction_method", "language", "domain",
    ):
        assert key in meta
    assert meta["section_path"] == "Internet > Renewal"
    assert "raw_html" not in meta


def test_empty_section_produces_no_chunks(document, lexical_embeddings):
    chunker = SemanticChunker(lexical_embeddings, ChunkingConfig.from_settings())
    assert chunker.chunk_document(document, [_section("   ")]) == []


def test_cosine_bounds():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)
    assert cosine([0, 0], [1, 0]) == 0.0
