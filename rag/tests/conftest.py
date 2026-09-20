import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from rag.embeddings.local_providers import HashEmbedding, LocalLexicalEmbedding
from rag.models import Document


@pytest.fixture
def hash_embeddings():
    return HashEmbedding()


@pytest.fixture
def lexical_embeddings():
    return LocalLexicalEmbedding()


@pytest.fixture
def document():
    return Document(
        document_id="doc1",
        source_url="https://web.vodafone.com.eg/en/test",
        domain="web.vodafone.com.eg",
        raw_html=None,
        raw_text="body",
        llm_enhanced_text=None,
        retrieval_text="body",
        retrieval_text_source="raw_text",
        language="en",
        scrape_status="success",
        scraped_at=None,
        dataset_file="batch_0000.parquet",
    )
