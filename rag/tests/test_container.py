"""Composition-root regression tests."""

import pytest

from rag import container


class FakeEmbeddings:
    def __init__(self, model: str, dimension: int):
        self.model = model
        self._dimension = dimension

    @property
    def dimension(self):
        return self._dimension


class FakeStore:
    collection_name = "kb_company_1"

    def __init__(self, dimension):
        self._dimension = dimension
        self.embedding_model = "configured"

    def embedding_dimension(self):
        return self._dimension


def test_empty_collection_keeps_configured_embeddings():
    configured = FakeEmbeddings("BAAI/bge-m3", 1024)
    assert container._resolve_collection_embeddings(
        configured, FakeStore(None)
    ) is configured


def test_matching_collection_keeps_configured_embeddings():
    configured = FakeEmbeddings("BAAI/bge-m3", 1024)
    assert container._resolve_collection_embeddings(
        configured, FakeStore(1024)
    ) is configured


def test_legacy_collection_selects_compatible_embeddings(monkeypatch):
    configured = FakeEmbeddings("BAAI/bge-m3", 1024)
    legacy = FakeEmbeddings("google/embeddinggemma-300m", 768)
    store = FakeStore(768)

    def fake_builder(provider=None, model=None, **_):
        assert provider == "embeddinggemma_hf"
        assert model == "google/embeddinggemma-300m"
        return legacy

    monkeypatch.setattr(container, "build_embedding_service", fake_builder)
    selected = container._resolve_collection_embeddings(configured, store)

    assert selected is legacy
    assert store.embedding_model == legacy.model


def test_unknown_collection_dimension_fails_during_wiring():
    configured = FakeEmbeddings("BAAI/bge-m3", 1024)
    with pytest.raises(RuntimeError, match="dimension 384"):
        container._resolve_collection_embeddings(configured, FakeStore(384))
