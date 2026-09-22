"""Chroma Cloud store behaviour, exercised against a mocked Chroma client.

No live Chroma Cloud credentials are required.
"""

import pytest

from tests.factories import make_test_chunk
from rag.vectorstore.chroma_cloud import ChromaCloudStore, build_where, parse_query_result
from rag.vectorstore.memory_store import InMemoryVectorStore


class FakeCollection:
    """Minimal stand-in for a chromadb Collection."""

    def __init__(self):
        self.store: dict[str, dict] = {}
        self.calls: list[str] = []

    def add(self, ids, documents, metadatas, embeddings):
        self.calls.append("add")
        for i, cid in enumerate(ids):
            if cid in self.store:
                raise ValueError(f"duplicate id {cid}")
            self.store[cid] = {
                "document": documents[i], "metadata": metadatas[i], "embedding": embeddings[i]
            }

    def upsert(self, ids, documents, metadatas, embeddings):
        self.calls.append("upsert")
        for i, cid in enumerate(ids):
            self.store[cid] = {
                "document": documents[i], "metadata": metadatas[i], "embedding": embeddings[i]
            }

    def delete(self, ids):
        self.calls.append("delete")
        for cid in ids:
            self.store.pop(cid, None)

    def get(self, where=None, include=None):
        self.calls.append("get")
        key, cond = next(iter(where.items()))
        wanted = cond["$eq"] if isinstance(cond, dict) else cond
        return {"ids": [c for c, r in self.store.items() if r["metadata"].get(key) == wanted]}

    def query(self, query_embeddings, n_results, where=None, include=None):
        self.calls.append("query")
        items = list(self.store.items())[:n_results]
        return {
            "ids": [[c for c, _ in items]],
            "documents": [[r["document"] for _, r in items]],
            "metadatas": [[r["metadata"] for _, r in items]],
            "distances": [[0.1 * (i + 1) for i in range(len(items))]],
        }

    def count(self):
        return len(self.store)


class FakeClient:
    def __init__(self):
        self.collection = FakeCollection()

    def get_or_create_collection(self, name, metadata=None):
        return self.collection


def make_chunk(i=0, document_id="doc1", text="some chunk text", **kw):
    return make_test_chunk(part=i, text=text, document_id=document_id, **kw)


@pytest.fixture
def store():
    return ChromaCloudStore(collection_name="test", client=FakeClient(), embedding_model="m")


def test_add_and_count(store):
    store.add_chunks([make_chunk(0), make_chunk(1)], [[0.1, 0.2], [0.3, 0.4]])
    assert store.count() == 2


def test_upsert_prevents_duplicates(store):
    chunks, vectors = [make_chunk(0)], [[0.1, 0.2]]
    store.upsert_chunks(chunks, vectors)
    store.upsert_chunks(chunks, vectors)  # identical deterministic id
    assert store.count() == 1


def test_add_rejects_duplicate_ids(store):
    store.add_chunks([make_chunk(0)], [[0.1, 0.2]])
    with pytest.raises(ValueError):
        store.add_chunks([make_chunk(0)], [[0.1, 0.2]])


def test_mismatched_lengths_raise(store):
    with pytest.raises(ValueError, match="mismatch"):
        store.upsert_chunks([make_chunk(0), make_chunk(1)], [[0.1]])


def test_delete_document_and_reingest(store):
    store.upsert_chunks([make_chunk(0), make_chunk(1)], [[0.1], [0.2]])
    assert store.delete_document("doc1") == 2
    assert store.count() == 0
    store.upsert_chunks([make_chunk(0)], [[0.1]])
    assert store.count() == 1


def test_delete_chunks_by_id(store):
    store.upsert_chunks([make_chunk(0), make_chunk(1)], [[0.1], [0.2]])
    removed = store.delete_chunks([make_chunk(0).chunk_id])
    assert removed == 1 and store.count() == 1


def test_metadata_is_preserved_and_html_free(store):
    store.upsert_chunks([make_chunk(0)], [[0.1, 0.2]])
    results = store.similarity_search([0.1, 0.2], top_k=1)
    meta = results[0].metadata
    assert meta["section_path"] == "Internet > Renewal"
    assert meta["document_id"] == "doc1"
    assert "raw_html" not in meta


def test_similarity_search_maps_distance_to_score(store):
    store.upsert_chunks([make_chunk(0)], [[0.1, 0.2]])
    result = store.similarity_search([0.1, 0.2], top_k=1)[0]
    assert result.distance == pytest.approx(0.1)
    assert result.score == pytest.approx(0.9)
    assert result.section_path == ["Internet", "Renewal"]


def test_info_never_exposes_the_api_key(store):
    info = store.info()
    assert info["api_key"] in ("<set>", "<unset>")
    assert "CHROMA_API_KEY" not in str(info)


def test_missing_credentials_raise_a_clear_error(monkeypatch):
    from rag.config import settings

    monkeypatch.setattr(settings, "CHROMA_API_KEY", "")
    with pytest.raises(RuntimeError, match="not configured"):
        ChromaCloudStore._build_client()


# ---- where-clause translation --------------------------------------------
def test_build_where_single_and_multiple():
    assert build_where({"domain": "x"}) == {"domain": {"$eq": "x"}}
    clause = build_where({"domain": "x", "language": "ar"})
    assert "$and" in clause and len(clause["$and"]) == 2


def test_build_where_handles_lists_and_none():
    assert build_where({"language": ["ar", "en"]}) == {"language": {"$in": ["ar", "en"]}}
    assert build_where({"domain": None}) == {}
    assert build_where(None) == {}


def test_parse_empty_query_result():
    assert parse_query_result({"ids": [[]]}) == []


# ---- in-memory store parity ----------------------------------------------
def test_memory_store_filters_by_metadata():
    store = InMemoryVectorStore()
    a, b = make_chunk(0, "doc1"), make_chunk(1, "doc2")
    b.language = "ar"
    store.upsert_chunks([a, b], [[1.0, 0.0], [0.0, 1.0]])
    results = store.similarity_search([1.0, 0.0], top_k=5, filters={"language": "ar"})
    assert [r.document_id for r in results] == ["doc2"]


def test_memory_store_upsert_is_idempotent():
    store = InMemoryVectorStore()
    for _ in range(3):
        store.upsert_chunks([make_chunk(0)], [[1.0, 0.0]])
    assert store.count() == 1
