"""Document normalisation, retrieval-text policy, and idempotent ingestion."""

import pytest

from rag.ingestion.document_builder import build_document, is_usable, select_retrieval_text
from rag.embeddings.local_providers import HashEmbedding
from rag.services.ingest_service import IngestionPipeline
from rag.text_utils import detect_language, normalize_text, same_script
from rag.vectorstore.memory_store import InMemoryVectorStore

HTML = (
    "<html><body><main><h1>Internet</h1>"
    "<h2>Renewal</h2><p>You can renew your bundle by dialling *880# from your line. "
    "The bundle costs 70 EGP per month and renews automatically every cycle.</p>"
    "</main></body></html>"
)


def row(**overrides):
    base = {
        "source_url": "https://web.vodafone.com.eg/en/flex",
        "scrape_status": "success",
        "raw_html": HTML,
        "raw_text": "Internet Renewal You can renew your bundle by dialling *880#.",
        "llm_enhanced_text": "Renew your Vodafone bundle by dialling *880# for 70 EGP.",
        "scraped_at": None,
        "dataset_file": "batch_0000.parquet",
    }
    base.update(overrides)
    return base


# ---- filtering ------------------------------------------------------------
def test_only_successful_rows_are_usable():
    assert is_usable(row())
    assert not is_usable(row(scrape_status="failed:timeout"))
    assert not is_usable(row(scrape_status="success", raw_html=None, raw_text=None,
                             llm_enhanced_text=None))


def test_nan_values_are_treated_as_missing():
    document = build_document(row(llm_enhanced_text=float("nan")))
    assert document.llm_enhanced_text is None
    assert document.retrieval_text_source == "raw_text"


def test_document_id_is_stable_across_runs():
    assert build_document(row()).document_id == build_document(row()).document_id


def test_domain_and_dataset_file_are_preserved():
    document = build_document(row())
    assert document.domain == "web.vodafone.com.eg"
    assert document.dataset_file == "batch_0000.parquet"
    assert document.raw_html and document.raw_text


# ---- retrieval-text policy ------------------------------------------------
def test_enhanced_first_prefers_enhanced_text():
    text, source = select_retrieval_text("enhanced", "raw", policy="enhanced_first")
    assert (text, source) == ("enhanced", "llm_enhanced_text")


def test_falls_back_to_raw_when_enhanced_missing():
    text, source = select_retrieval_text("", "raw", policy="enhanced_first")
    assert (text, source) == ("raw", "raw_text")


def test_language_aware_policy_keeps_arabic_pages_arabic():
    arabic = "ازاي أجدد باقة الإنترنت من تطبيق أنا فودافون بكل سهولة وسرعة"
    english = "Renew your internet bundle through the Ana Vodafone application."
    text, source = select_retrieval_text(english, arabic, policy="language_aware")
    assert source == "raw_text" and text == arabic
    # Same-script pair still prefers the enhanced version.
    text2, source2 = select_retrieval_text(english, english, policy="language_aware")
    assert source2 == "llm_enhanced_text"


def test_raw_only_policy():
    assert select_retrieval_text("enhanced", "raw", policy="raw_only")[1] == "raw_text"


# ---- language utilities ---------------------------------------------------
def test_language_detection():
    assert detect_language("ازاي أجدد باقة الإنترنت؟") == "ar"
    assert detect_language("How do I renew my bundle?") == "en"
    assert detect_language("ازاي أجدد الـ Internet bundle بتاعي؟") == "mixed"
    assert detect_language("") == "unknown"


def test_same_script_comparison():
    assert same_script("مرحبا بك", "أهلا بك")
    assert not same_script("مرحبا بك", "Hello there")


def test_normalize_strips_diacritics_and_collapses_whitespace():
    assert normalize_text("  مَرْحَبــا   بك  ") == "مرحبا بك"
    assert normalize_text("a\n\n\n\nb") == "a\n\nb"


# ---- pipeline idempotency -------------------------------------------------
@pytest.fixture
def pipeline():
    embeddings = HashEmbedding()
    return IngestionPipeline(InMemoryVectorStore(), embeddings), embeddings


def test_chunks_are_built_from_html_sections(pipeline):
    pipe, _ = pipeline
    sections, chunks = pipe.chunks_for(build_document(row()))
    assert sections and chunks
    assert chunks[0].section_extraction_method == "html_structure"
    assert chunks[0].section_path == ["Internet", "Renewal"]
    # HTML is never used as the embedding text.
    assert "<p>" not in chunks[0].text


def test_reingestion_does_not_duplicate_vectors(pipeline, tmp_path):
    pipe, _ = pipeline
    document = build_document(row())
    _, chunks = pipe.chunks_for(document)
    vectors = pipe.embeddings.embed_documents([c.text for c in chunks])

    pipe.store.upsert_chunks(chunks, vectors)
    first = pipe.store.count()
    pipe.store.upsert_chunks(chunks, vectors)
    assert pipe.store.count() == first


def test_changed_document_prunes_stale_chunks(pipeline):
    pipe, _ = pipeline
    document = build_document(row())
    _, chunks = pipe.chunks_for(document)
    pipe.store.upsert_chunks(chunks, pipe.embeddings.embed_documents([c.text for c in chunks]))
    before = pipe.store.count()

    changed = build_document(row(raw_html=HTML.replace("70 EGP", "95 EGP")))
    _, new_chunks = pipe.chunks_for(changed)
    removed = pipe._prune_stale(changed, new_chunks)
    pipe.store.upsert_chunks(
        new_chunks, pipe.embeddings.embed_documents([c.text for c in new_chunks])
    )
    assert removed > 0
    assert pipe.store.count() == before  # replaced, not accumulated
    assert "95 EGP" in " ".join(
        r["text"] for r in pipe.store._docs.values()
    )


def test_delete_document_enables_clean_reingest(pipeline):
    pipe, _ = pipeline
    document = build_document(row())
    _, chunks = pipe.chunks_for(document)
    pipe.store.upsert_chunks(chunks, pipe.embeddings.embed_documents([c.text for c in chunks]))
    assert pipe.store.delete_document(document.document_id) > 0
    assert pipe.store.count() == 0


# ---- soft-404 filtering ---------------------------------------------------
def test_soft_404_pages_are_excluded():
    """~15% of 'success' rows in this corpus render the not-found template."""
    not_found = row(
        raw_text="notfound الصفحة غير موجودة! لا يوجد صفحة مسجلة بهذا العنوان.",
        llm_enhanced_text=None,
    )
    assert not is_usable(not_found)
    assert build_document(not_found) is None


def test_english_soft_404_is_excluded():
    assert not is_usable(row(raw_text="Page Not Found. Sorry.", llm_enhanced_text=None))


def test_real_page_mentioning_not_found_later_is_kept():
    """The marker is only checked in the opening of the page."""
    body = "Renew your bundle here. " * 80 + " page not found"
    assert is_usable(row(raw_text=body))


# ---- cross-document sentence batching -------------------------------------
def test_chunk_documents_embeds_all_sentences_in_one_pass(document):
    """Sentences must be batched across documents, not one call per document."""
    from rag.chunking.semantic_chunker import SemanticChunker
    from rag.embeddings.base import EmbeddingService

    class CountingEmbeddings(EmbeddingService):
        model = "counting"

        def __init__(self):
            self.calls = 0
            self.total_texts = 0

        @property
        def dimension(self):
            return 8

        def embed_documents(self, texts):
            self.calls += 1
            self.total_texts += len(texts)
            return [[float(len(t) % 7)] * 8 for t in texts]

        def embed_query(self, text):
            return [1.0] * 8

    from rag.models import Section

    def section(i):
        return Section(
            section_index=0, section_title=f"S{i}", section_path=[f"S{i}"],
            heading_level=2,
            text=f"Sentence one of section {i}. Sentence two of section {i}. "
                 f"Sentence three of section {i}.",
            extraction_method="html_structure",
        )

    embeddings = CountingEmbeddings()
    chunker = SemanticChunker(embeddings)
    pairs = [(document, [section(i)]) for i in range(5)]
    results = chunker.chunk_documents(pairs)

    assert len(results) == 5
    assert embeddings.calls == 1, "five documents must cost one embedding call"
    assert embeddings.total_texts == 15


def test_chunk_document_still_works_for_a_single_document(document, lexical_embeddings):
    from rag.chunking.semantic_chunker import SemanticChunker
    from rag.models import Section

    section = Section(
        section_index=0, section_title="S", section_path=["S"], heading_level=2,
        text="Renew your bundle from the app. It takes a minute.",
        extraction_method="html_structure",
    )
    chunks = SemanticChunker(lexical_embeddings).chunk_document(document, [section])
    assert chunks and chunks[0].section_title == "S"


def test_chunking_embedding_provider_override(monkeypatch):
    """CHUNKING_EMBEDDING_PROVIDER decouples boundary vectors from stored ones."""
    from rag.config import settings
    from rag.services.ingest_service import _chunking_embeddings
    from rag.embeddings.local_providers import HashEmbedding

    default = HashEmbedding()
    monkeypatch.setattr(settings, "CHUNKING_EMBEDDING_PROVIDER", "")
    assert _chunking_embeddings(default) is default

    monkeypatch.setattr(settings, "CHUNKING_EMBEDDING_PROVIDER", "local_lexical")
    swapped = _chunking_embeddings(default)
    assert swapped is not default
    assert "lexical" in swapped.model


def test_pruning_is_skipped_on_an_empty_collection(pipeline):
    pipe, _ = pipeline
    stats = pipe.run(limit=2, dry_run=True)
    assert stats.rows_read == 2
