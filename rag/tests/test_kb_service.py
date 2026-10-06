import pytest

from rag.models import Chunk
from rag.services.kb_service import _deduplicate_semantic_chunks


def make_test_chunk(part: int, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"1:doc1:{part}:h{part}",
        company_id=1,
        text=text,
        source="manual.pdf",
        path="/kb/1/manual.pdf",
        part=part,
        document_id="doc1",
        section_title="Renewal",
        section_path=["Internet", "Renewal"],
        heading_level=2,
        section_index=0,
        chunk_index=part - 1,
        section_extraction_method="pdf_font",
        page_number=1,
        language="en",
    )


def test_semantic_chunk_deduplication_keeps_first_similar_chunk():
    chunks = [
        make_test_chunk(part=1, text="Renew the bundle in the mobile app."),
        make_test_chunk(part=2, text="Use the mobile application to renew the bundle."),
        make_test_chunk(part=3, text="Update the billing address in the account portal."),
    ]
    vectors = [
        [1.0, 0.0],
        [0.99, 0.1],
        [0.0, 1.0],
    ]

    unique, unique_vectors = _deduplicate_semantic_chunks(
        chunks, vectors, threshold=0.95
    )

    assert [chunk.text for chunk in unique] == [
        "Renew the bundle in the mobile app.",
        "Update the billing address in the account portal.",
    ]
    assert unique_vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert [chunk.part for chunk in unique] == [1, 2]
    assert [chunk.chunk_index for chunk in unique] == [0, 1]
    assert unique[0].chunk_id != unique[1].chunk_id


def test_semantic_chunk_deduplication_rejects_invalid_threshold():
    with pytest.raises(ValueError, match="threshold"):
        _deduplicate_semantic_chunks([], [], threshold=1.1)
