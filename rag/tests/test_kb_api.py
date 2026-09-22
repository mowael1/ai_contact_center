"""Knowledge-base API: upload, list, delete, ask - and tenant isolation.

The vector store is swapped per company so the routes exercise the same
collection-per-tenant behaviour they use against Chroma Cloud, with no network.
"""

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from rag.api.app import create_app
from rag.api.deps import get_kb_container, get_tenant
from rag.container import Container
from rag.embeddings.local_providers import LocalLexicalEmbedding
from rag.llm.base import LLMResponse, LLMService
from rag.models import LLMUsage
from rag.services.retriever import RetrievalService
from rag.services.tenancy import Tenant
from rag.vectorstore.memory_store import InMemoryVectorStore

COMPANY_A_DOC = (
    "# Returns and refunds\n\n"
    "If you return an item bought via installments you receive a refund "
    "credited to your noon Credits account, not as cash.\n\n"
    "# Minimum order\n\nThe minimum eligible order is 500 EGP.\n"
)
COMPANY_B_DOC = (
    "# Supervisor escalation\n\n"
    "Refunds above 5000 EGP require supervisor approval via the internal portal.\n"
)


class StubLLM(LLMService):
    model = "stub"

    def generate(self, system, prompt, max_tokens=1024, temperature=0.0):
        # Echo enough of the context that tests can assert grounding.
        return LLMResponse("Answer based on the documents [1].",
                           LLMUsage(100, 20, self.model, 0.001))


@pytest.fixture(autouse=True)
def dev_mode(monkeypatch):
    """These tests address companies by header, which is dev mode."""
    from rag.config import settings

    monkeypatch.setattr(settings, "RAG_AUTH_MODE", "dev")


@pytest.fixture
def client():
    """One in-memory collection per company, resolved from X-Company-Id."""
    embeddings = LocalLexicalEmbedding()
    stores: dict[int, InMemoryVectorStore] = {}
    app = create_app()

    # A container per company, resolved from the same get_tenant dependency
    # the real routes use.
    def get_container_override(tenant: Tenant = Depends(get_tenant)):
        store = stores.setdefault(
            tenant.company_id,
            InMemoryVectorStore(tenant.collection_name),
        )
        return Container(
            embeddings=embeddings,
            store=store,
            retriever=RetrievalService(store, embeddings, tenant=tenant),
            tenant=tenant,
            _llm=StubLLM(),
        )

    app.dependency_overrides[get_kb_container] = get_container_override
    return TestClient(app)


def upload(client, company_id: int, name: str, body: str):
    return client.post(
        "/api/v1/kb/documents",
        headers={"X-Company-Id": str(company_id)},
        files={"file": (name, body.encode("utf-8"), "text/markdown")},
    )


# ---- upload ---------------------------------------------------------------
def test_upload_indexes_a_document(client):
    response = upload(client, 1, "returns.md", COMPANY_A_DOC)
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "ingested"
    assert body["company_id"] == 1
    assert body["collection"] == "kb_company_1"
    assert body["chunks"] >= 2
    assert body["strategy"] == "section_based"


def test_upload_rejects_unsupported_types(client):
    response = client.post(
        "/api/v1/kb/documents",
        headers={"X-Company-Id": "1"},
        files={"file": ("photo.png", b"\x89PNG", "image/png")},
    )
    assert response.status_code == 415


def test_upload_is_idempotent(client):
    upload(client, 1, "returns.md", COMPANY_A_DOC)
    first = client.get("/api/v1/kb/documents", headers={"X-Company-Id": "1"}).json()
    upload(client, 1, "returns.md", COMPANY_A_DOC)
    second = client.get("/api/v1/kb/documents", headers={"X-Company-Id": "1"}).json()
    assert second["vectors"] == first["vectors"]
    assert len(second["documents"]) == 1


# ---- listing and deletion -------------------------------------------------
def test_list_returns_only_own_documents(client):
    upload(client, 1, "returns.md", COMPANY_A_DOC)
    upload(client, 2, "escalation.md", COMPANY_B_DOC)

    a = client.get("/api/v1/kb/documents", headers={"X-Company-Id": "1"}).json()
    b = client.get("/api/v1/kb/documents", headers={"X-Company-Id": "2"}).json()
    assert [d["source"] for d in a["documents"]] == ["returns.md"]
    assert [d["source"] for d in b["documents"]] == ["escalation.md"]
    assert a["collection"] != b["collection"]


def test_cannot_delete_another_companys_document(client):
    uploaded = upload(client, 1, "returns.md", COMPANY_A_DOC).json()
    document_id = uploaded["document_id"]

    denied = client.delete(
        f"/api/v1/kb/documents/{document_id}", headers={"X-Company-Id": "2"}
    )
    assert denied.status_code == 404

    still_there = client.get("/api/v1/kb/documents", headers={"X-Company-Id": "1"}).json()
    assert len(still_there["documents"]) == 1


def test_owner_can_delete_its_own_document(client):
    uploaded = upload(client, 1, "returns.md", COMPANY_A_DOC).json()
    response = client.delete(
        f"/api/v1/kb/documents/{uploaded['document_id']}",
        headers={"X-Company-Id": "1"},
    )
    assert response.status_code == 200
    assert response.json()["deleted_chunks"] == uploaded["chunks"]


# ---- ask and isolation ----------------------------------------------------
def test_ask_answers_from_own_documents(client):
    upload(client, 1, "returns.md", COMPANY_A_DOC)
    body = client.post(
        "/api/v1/kb/ask",
        headers={"X-Company-Id": "1"},
        json={"query": "How do refunds work for installments?", "top_k": 3},
    ).json()
    assert body["has_sufficient_context"] is True
    assert body["citations"]
    assert body["citations"][0]["source"] == "returns.md"
    assert "page" in body["citations"][0]["label"] or body["citations"][0]["label"]


def test_ask_cannot_reach_another_companys_documents(client):
    upload(client, 2, "escalation.md", COMPANY_B_DOC)
    body = client.post(
        "/api/v1/kb/ask",
        headers={"X-Company-Id": "1"},
        json={"query": "Who approves refunds above 5000 EGP?", "top_k": 5},
    ).json()
    assert body["has_sufficient_context"] is False
    assert body["citations"] == []


def test_ask_includes_chunks_on_request(client):
    upload(client, 1, "returns.md", COMPANY_A_DOC)
    body = client.post(
        "/api/v1/kb/ask",
        headers={"X-Company-Id": "1"},
        json={"query": "refunds", "top_k": 2, "include_chunks": True},
    ).json()
    assert body["chunks"]
    assert all(c["metadata"]["company_id"] == 1 for c in body["chunks"])


def test_ask_rejects_an_empty_query(client):
    response = client.post(
        "/api/v1/kb/ask", headers={"X-Company-Id": "1"}, json={"query": ""}
    )
    assert response.status_code == 422


# ---- info -----------------------------------------------------------------
def test_info_reports_the_scoped_collection(client):
    body = client.get("/api/v1/kb/info", headers={"X-Company-Id": "9"}).json()
    assert body["company_id"] == 9
    assert body["collection"] == "kb_company_9"
    assert "chunking" in body


def test_a_request_without_a_company_is_refused(client):
    """No scope must be an error.

    This previously defaulted to a fixed company, so every admin resolved to
    the same tenant and saw one company's documents.
    """
    assert client.get("/api/v1/kb/info").status_code == 400


# ---- conversation memory --------------------------------------------------
def test_follow_up_is_expanded_for_retrieval(client):
    upload(client, 1, "returns.md", COMPANY_A_DOC)
    headers = {"X-Company-Id": "1"}
    client.post("/api/v1/kb/ask", headers=headers,
                json={"query": "What is the refund policy?", "conversation_id": "c1"})
    body = client.post("/api/v1/kb/ask", headers=headers,
                       json={"query": "وكام مدته؟", "conversation_id": "c1"}).json()
    assert body["search_query"], "a follow-up should be expanded"
    assert "refund" in body["search_query"].lower()


def test_a_standalone_question_is_not_expanded(client):
    upload(client, 1, "returns.md", COMPANY_A_DOC)
    body = client.post(
        "/api/v1/kb/ask", headers={"X-Company-Id": "1"},
        json={"query": "What is the minimum eligible order value?", "conversation_id": "c2"},
    ).json()
    assert body["search_query"] is None


def test_conversations_do_not_leak_between_companies(client):
    """The same conversation id under two companies must stay separate."""
    upload(client, 1, "returns.md", COMPANY_A_DOC)
    upload(client, 2, "escalation.md", COMPANY_B_DOC)
    client.post("/api/v1/kb/ask", headers={"X-Company-Id": "1"},
                json={"query": "What is the refund policy?", "conversation_id": "shared"})
    body = client.post("/api/v1/kb/ask", headers={"X-Company-Id": "2"},
                       json={"query": "وكام مدته؟", "conversation_id": "shared"}).json()
    # Company 2 has no history under that id, so nothing is carried over.
    assert body["search_query"] is None


def test_clearing_a_conversation_forgets_it(client):
    upload(client, 1, "returns.md", COMPANY_A_DOC)
    headers = {"X-Company-Id": "1"}
    client.post("/api/v1/kb/ask", headers=headers,
                json={"query": "What is the refund policy?", "conversation_id": "c3"})
    assert client.delete("/api/v1/kb/conversations/c3", headers=headers).json()["cleared"]
    body = client.post("/api/v1/kb/ask", headers=headers,
                       json={"query": "وكام مدته؟", "conversation_id": "c3"}).json()
    assert body["search_query"] is None


def test_a_question_without_a_conversation_id_is_not_remembered(client):
    upload(client, 1, "returns.md", COMPANY_A_DOC)
    headers = {"X-Company-Id": "1"}
    client.post("/api/v1/kb/ask", headers=headers, json={"query": "What is the refund policy?"})
    body = client.post("/api/v1/kb/ask", headers=headers,
                       json={"query": "وكام مدته؟"}).json()
    assert body["search_query"] is None
