"""API contract tests. No live Chroma Cloud and no real LLM are used."""

import pytest
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.container import Container, get_container
from rag.embeddings.local_providers import HashEmbedding
from rag.llm.base import LLMResponse, LLMService
from rag.models import Chunk, LLMUsage
from rag.services.retriever import RetrievalService
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


def make_chunk(i, text, url="https://web.vodafone.com.eg/en/flex"):
    return Chunk(
        chunk_id=f"doc1:0:{i}:h{i}", document_id="doc1", source_url=url, text=text,
        section_title="Renewal", section_path=["Internet", "Renewal"], heading_level=3,
        section_index=0, chunk_index=i, chunking_method="semantic",
        section_extraction_method="html_structure", language="en",
        domain="web.vodafone.com.eg",
    )


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
        retriever=RetrievalService(store, embeddings), _llm=llm,
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
    assert first["source_url"].startswith("https://")
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
    assert body["citations"][0]["source_url"].startswith("https://")
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
        retriever=RetrievalService(store, embeddings), _llm=StubLLM(),
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
        retriever=RetrievalService(store, embeddings),
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
        "/api/v1/rag/query/stream",
    }


def test_query_runs_the_agentic_loop_by_default(client):
    """AGENTIC_ENABLED defaults true, so /query reports its audit trail."""
    body = client.post("/api/v1/rag/query", json={"query": "renew", "top_k": 1}).json()
    assert body["attempts"] and body["attempts"][0]["is_relevant"] is True
    assert any(t.startswith("retrieve") for t in body["trace"])


# ==========================================================================
# Streaming + agentic endpoints
# ==========================================================================
import json as _json

from rag.llm.base import StreamChunk


class StreamingStubLLM(LLMService):
    model = "stream-stub"

    def __init__(self, text="You can renew it from the app [1]."):
        self.text = text
        self.stream_calls = 0

    def generate(self, system, prompt, max_tokens=1024, temperature=0.0):
        return LLMResponse(self.text, LLMUsage(120, 30, self.model, 0.001))

    def generate_stream(self, system, prompt, max_tokens=1024, temperature=0.0):
        self.stream_calls += 1
        accumulated = ""
        for word in self.text.split(" "):
            accumulated += (" " if accumulated else "") + word
            yield StreamChunk(
                delta=(" " if accumulated != word else "") + word, text=accumulated
            )
        yield StreamChunk(
            text=accumulated, done=True, usage=LLMUsage(120, 30, self.model, 0.001)
        )


def parse_sse(body: str):
    """Return [(event, data_dict), ...] from an SSE response body."""
    events = []
    for block in body.strip().split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data = _json.loads(line[len("data:"):].strip())
        if name:
            events.append((name, data))
    return events


@pytest.fixture
def streaming_client():
    embeddings = HashEmbedding()
    store = InMemoryVectorStore(embedding_model=embeddings.model)
    chunks = [make_chunk(0, "You can renew your bundle through Ana Vodafone or dial *880#.")]
    store.upsert_chunks(chunks, embeddings.embed_documents([c.text for c in chunks]))
    llm = StreamingStubLLM()
    container = Container(
        embeddings=embeddings, store=store,
        retriever=RetrievalService(store, embeddings), _llm=llm,
        _judge_llm=RelevantJudge(),
    )
    app = create_app()
    app.dependency_overrides[get_container] = lambda: container
    return TestClient(app), llm


def test_stream_endpoint_emits_deltas_then_done(streaming_client):
    client, _ = streaming_client
    response = client.post(
        "/api/v1/rag/query/stream",
        json={"query": "How do I renew?", "top_k": 1, "agentic": False},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[-1] == "done"
    assert "delta" in names


def test_stream_deltas_reconstruct_the_final_answer(streaming_client):
    client, _ = streaming_client
    response = client.post(
        "/api/v1/rag/query/stream", json={"query": "renew", "agentic": False}
    )
    rebuilt = ""
    final = None
    for name, data in parse_sse(response.text):
        if name == "delta":
            if data.get("replaces_from") is not None:
                rebuilt = rebuilt[: data["replaces_from"]]
            rebuilt += data["delta"]
        elif name == "done":
            final = data
    assert final is not None
    assert rebuilt == final["answer"]


def test_stream_sends_citations_only_on_done(streaming_client):
    client, _ = streaming_client
    response = client.post(
        "/api/v1/rag/query/stream", json={"query": "renew", "agentic": False}
    )
    for name, data in parse_sse(response.text):
        if name == "delta":
            assert "citations" not in data
        elif name == "done":
            assert data["citations"]
            assert data["citations"][0]["source_url"].startswith("https://")


def test_stream_actually_uses_the_streaming_path(streaming_client):
    client, llm = streaming_client
    client.post("/api/v1/rag/query/stream", json={"query": "renew", "agentic": False})
    assert llm.stream_calls == 1


def test_query_endpoint_accepts_agentic_flag(client):
    body = client.post(
        "/api/v1/rag/query", json={"query": "renew", "top_k": 1, "agentic": False}
    ).json()
    assert body["attempts"] == []
    assert body["has_sufficient_context"] is True


def test_openapi_documents_the_stream_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert "/api/v1/rag/query/stream" in paths
