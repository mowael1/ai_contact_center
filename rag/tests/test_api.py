"""API contract tests. No live Chroma Cloud and no real LLM are used."""

import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.container import Container, get_container
from rag.embeddings.local_providers import HashEmbedding
from rag.llm.base import LLMResponse, LLMService
from tests.factories import make_test_chunk
from rag.models import LLMUsage
from rag.services.retriever import RetrievalService
from rag.services.tenancy import Tenant
from rag.vectorstore.memory_store import InMemoryVectorStore


class RelevantJudge(LLMService):
    """Judge stub that always finds the context sufficient."""

    model = "judge-stub"

    def generate(self, system, prompt, max_tokens=1024, temperature=0.0):
        return LLMResponse(
            '{"is_relevant": true, "confidence": 0.95, "reason": "ok", "missing": ""}',
            LLMUsage(20, 10, self.model, 0.0),
        )


class StubLLM(LLMService):
    model = "stub"

    def __init__(self, text="You can renew it from the app [1]."):
        self.text = text
        self.prompts: list[str] = []

    def generate(self, system, prompt, max_tokens=1024, temperature=0.0):
        self.prompts.append(prompt)
        return LLMResponse(self.text, LLMUsage(120, 30, self.model, 0.001))


def make_chunk(i, text, **kw):
    return make_test_chunk(part=i + 1, text=text, **kw)


@pytest.fixture
def client_and_llm():
    embeddings = HashEmbedding()
    store = InMemoryVectorStore(embedding_model=embeddings.model)
    chunks = [
        make_chunk(0, "You can renew your bundle through Ana Vodafone app or dial *880#."),
        make_chunk(1, "Vodafone Cash gives 30% extra when paying for your bundle."),
    ]
    store.upsert_chunks(chunks, embeddings.embed_documents([c.text for c in chunks]))

    llm = StubLLM()
    container = Container(
        embeddings=embeddings, store=store,
        retriever=RetrievalService(store, embeddings, tenant=Tenant(company_id=1)), _llm=llm,
        _judge_llm=RelevantJudge(),
    )
    app = create_app()
    app.dependency_overrides[get_container] = lambda: container
    return TestClient(app), llm


@pytest.fixture
def client(client_and_llm):
    return client_and_llm[0]


def test_health_reports_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["vector_store"]["vector_count"] == 2
    assert "api_key" not in str(body).lower() or "CHROMA_API_KEY" not in str(body)


def test_retrieve_returns_chunks_with_metadata(client):
    response = client.post("/api/v1/rag/retrieve", json={"query": "renew bundle", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    first = body["results"][0]
    assert first["section_path"] == ["Internet", "Renewal"]
    assert first["metadata"]["source"] == "manual.pdf"
    assert first["metadata"]["page_number"] == 1
    assert 0.0 <= first["score"] <= 1.0
    assert "total_ms" in body["latency_ms"]


def test_retrieve_respects_top_k(client):
    body = client.post("/api/v1/rag/retrieve", json={"query": "x", "top_k": 1}).json()
    assert body["count"] == 1


def test_retrieve_rejects_empty_query(client):
    assert client.post("/api/v1/rag/retrieve", json={"query": ""}).status_code == 422


def test_retrieve_filter_narrows_results(client):
    body = client.post(
        "/api/v1/rag/retrieve",
        json={"query": "x", "top_k": 5, "filters": {"language": "ar"}},
    ).json()
    assert body["count"] == 0


def test_query_returns_grounded_answer_and_citations(client):
    response = client.post("/api/v1/rag/query", json={"query": "How do I renew?", "top_k": 2})
    assert response.status_code == 200
    body = response.json()
    assert body["has_sufficient_context"] is True
    assert body["citations"]
    citation = body["citations"][0]
    assert citation["source"] == "manual.pdf"
    assert citation["page_number"] == 1
    assert "page 1" in citation["label"]
    assert body["usage"]["total_tokens"] == 150


def test_query_context_contains_retrieved_text(client_and_llm):
    client, llm = client_and_llm
    client.post("/api/v1/rag/query", json={"query": "How do I renew?"})
    assert "*880#" in llm.prompts[0]


def test_query_include_chunks_flag(client):
    body = client.post(
        "/api/v1/rag/query", json={"query": "renew", "top_k": 1, "include_chunks": True}
    ).json()
    assert body["chunks"] and len(body["chunks"]) == 1
    body2 = client.post("/api/v1/rag/query", json={"query": "renew", "top_k": 1}).json()
    assert body2["chunks"] is None


def test_query_marks_insufficient_context():
    embeddings = HashEmbedding()
    store = InMemoryVectorStore()  # empty
    container = Container(
        embeddings=embeddings, store=store,
        retriever=RetrievalService(store, embeddings, tenant=Tenant(company_id=1)), _llm=StubLLM(),
        _judge_llm=RelevantJudge(),
    )
    app = create_app()
    app.dependency_overrides[get_container] = lambda: container
    body = TestClient(app).post(
        "/api/v1/rag/query", json={"query": "anything", "agentic": False}
    ).json()
    assert body["has_sufficient_context"] is False
    assert body["citations"] == []


def test_fabricated_citation_index_is_dropped():
    """The model citing [9] when only 1 passage exists must not yield a citation."""
    embeddings = HashEmbedding()
    store = InMemoryVectorStore()
    chunk = make_chunk(0, "Renew via the app.")
    store.upsert_chunks([chunk], embeddings.embed_documents([chunk.text]))
    container = Container(
        embeddings=embeddings, store=store,
        retriever=RetrievalService(store, embeddings, tenant=Tenant(company_id=1)),
        _llm=StubLLM("Do it like this [9]."),
        _judge_llm=RelevantJudge(),
    )
    app = create_app()
    app.dependency_overrides[get_container] = lambda: container
    body = TestClient(app).post(
        "/api/v1/rag/query", json={"query": "renew", "top_k": 1}
    ).json()
    assert "[9]" not in body["answer"]
    assert all(c["index"] == 1 for c in body["citations"])


def test_openapi_documents_all_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) == {
        "/health",
        "/api/v1/rag/retrieve",
        "/api/v1/rag/query",
        "/api/v1/kb/documents",
        "/api/v1/kb/documents/{document_id}",
        "/api/v1/kb/ask",
        "/api/v1/kb/conversations/{conversation_id}",
        "/api/v1/kb/info",
    }


def test_query_runs_the_agentic_loop_by_default(client):
    """AGENTIC_ENABLED defaults true, so /query reports its audit trail."""
    body = client.post("/api/v1/rag/query", json={"query": "renew", "top_k": 1}).json()
    assert body["attempts"] and body["attempts"][0]["is_relevant"] is True
    assert any(t.startswith("retrieve") for t in body["trace"])


# ==========================================================================
# Agentic mode
# ==========================================================================
def test_query_endpoint_accepts_agentic_flag(client):
    body = client.post(
        "/api/v1/rag/query", json={"query": "renew", "top_k": 1, "agentic": False}
    ).json()
    assert body["attempts"] == []
    assert body["has_sufficient_context"] is True


def test_query_runs_the_agentic_loop_by_default(client):
    """AGENTIC_ENABLED defaults true, so /query reports its audit trail."""
    body = client.post("/api/v1/rag/query", json={"query": "renew", "top_k": 1}).json()
    assert body["attempts"] and body["attempts"][0]["is_relevant"] is True
    assert any(t.startswith("retrieve") for t in body["trace"])
