"""Re-ranker stage: Retriever -> Re-Ranker -> existing context builder.

No model is downloaded and no network is used - the cross-encoder is a scripted
fake injected into the retriever's per-process model cache.
"""

import json
import sys
import types

import pytest

import rag.services.retriever as retriever_module
from rag.config import settings
from rag.embeddings.local_providers import HashEmbedding
from rag.graph.workflow import AgenticRagWorkflow
from rag.llm.base import LLMResponse, LLMService
from rag.models import LLMUsage
from rag.services.evaluator import LLMContextEvaluator, LLMQueryRewriter
from rag.services.generator import GenerationService
from rag.services.rag_service import RagService
from rag.services.retriever import RetrievalService, rerank_chunks
from rag.services.tenancy import Tenant
from rag.vectorstore.memory_store import InMemoryVectorStore
from tests.factories import make_test_chunk

QUERY = "renewal topic filler"
TARGET_MARK = "GOLDEN-ANSWER"


class FakeCrossEncoder:
    """Scores 1.0 for passages containing ``favour``, else 0.0."""

    def __init__(self, favour=TARGET_MARK, boom=False, scores=None):
        self.favour = favour
        self.boom = boom
        self.scores = scores
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, pairs, show_progress_bar=False):
        self.calls.append(list(pairs))
        if self.boom:
            raise RuntimeError("inference exploded")
        if self.scores is not None:
            return self.scores(len(pairs))
        return [1.0 if self.favour in text else 0.0 for _, text in pairs]


@pytest.fixture(autouse=True)
def reranker_settings(monkeypatch):
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "TOP_K_RETRIEVAL", 10)
    monkeypatch.setattr(settings, "TOP_N_RERANK", 0)
    # Never let a test reach for a real model.
    monkeypatch.setattr(retriever_module, "_CROSS_ENCODERS", {})


def install(monkeypatch, fake):
    monkeypatch.setitem(retriever_module._CROSS_ENCODERS, settings.RERANK_MODEL, fake)
    return fake


def build_service(n=12, mark_index=None):
    embeddings = HashEmbedding()
    store = InMemoryVectorStore()
    chunks = []
    for i in range(n):
        text = f"renewal topic number {i} filler text about bundles"
        if i == mark_index:
            text += f" {TARGET_MARK}"
        chunks.append(
            make_test_chunk(
                part=i + 1, text=text, document_id=f"d{i}",
                source=f"file{i}.pdf", page_number=i + 1,
            )
        )
    if chunks:
        store.upsert_chunks(chunks, embeddings.embed_documents([c.text for c in chunks]))
    return RetrievalService(store, embeddings, tenant=Tenant(company_id=1))


def baseline_order(service, monkeypatch, top_k):
    """The retriever's own ranking with the re-ranker switched off."""
    monkeypatch.setattr(settings, "RERANK_ENABLED", False)
    ids = [c.chunk_id for c in service.retrieve(QUERY, top_k=top_k)]
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    return ids


def service_with_deep_target(monkeypatch):
    """A service whose marked chunk the plain retriever ranks outside the top 3
    but inside the 10-candidate pool, so only re-ranking can surface it."""
    for index in range(12):
        service = build_service(mark_index=index)
        order = baseline_order(service, monkeypatch, 12)
        target_id = f"1:d{index}:{index + 1}:h{index + 1}"
        if 3 <= order.index(target_id) < 10:
            return service, target_id
    pytest.fail("no chunk ranked between 4th and 10th in the baseline retrieval")


# 1 + 3 + 5 ----------------------------------------------------------------
def test_promotes_deep_candidate_and_keeps_only_top_n(monkeypatch):
    service, target_id = service_with_deep_target(monkeypatch)
    assert target_id not in baseline_order(service, monkeypatch, 3)
    fake = install(monkeypatch, FakeCrossEncoder())

    results = service.retrieve(QUERY, top_k=3)

    assert len(results) == 3
    assert results[0].chunk_id == target_id
    assert len(fake.calls) == 1


# 2 -------------------------------------------------------------------------
def test_reranker_receives_original_query_and_candidate_pool(monkeypatch):
    service, _ = service_with_deep_target(monkeypatch)
    fake = install(monkeypatch, FakeCrossEncoder())

    service.retrieve(QUERY, top_k=3)

    pairs = fake.calls[0]
    assert len(pairs) == 10  # TOP_K_RETRIEVAL candidates, not just top_k
    assert {q for q, _ in pairs} == {QUERY}
    assert all(isinstance(text, str) and text for _, text in pairs)


# 3 (configurable TOP_N) ----------------------------------------------------
def test_top_n_rerank_caps_but_never_exceeds_requested(monkeypatch):
    service, _ = service_with_deep_target(monkeypatch)
    install(monkeypatch, FakeCrossEncoder())

    monkeypatch.setattr(settings, "TOP_N_RERANK", 2)
    assert len(service.retrieve(QUERY, top_k=3)) == 2

    monkeypatch.setattr(settings, "TOP_N_RERANK", 8)
    assert len(service.retrieve(QUERY, top_k=3)) == 3


def test_top_k_retrieval_is_configurable(monkeypatch):
    service, _ = service_with_deep_target(monkeypatch)
    fake = install(monkeypatch, FakeCrossEncoder())
    monkeypatch.setattr(settings, "TOP_K_RETRIEVAL", 5)

    service.retrieve(QUERY, top_k=3)

    assert len(fake.calls[0]) == 5


def test_invalid_pool_setting_falls_back_to_requested_top_k(monkeypatch):
    service, _ = service_with_deep_target(monkeypatch)
    fake = install(monkeypatch, FakeCrossEncoder())
    monkeypatch.setattr(settings, "TOP_K_RETRIEVAL", -4)

    results = service.retrieve(QUERY, top_k=3)

    assert len(results) == 3
    assert len(fake.calls[0]) == 3


# 6 -------------------------------------------------------------------------
def test_objects_and_metadata_are_preserved(monkeypatch):
    service, target_id = service_with_deep_target(monkeypatch)
    plain = {c.chunk_id: c for c in service.retrieve(QUERY, top_k=12)}
    before = {
        cid: (c.text, c.score, c.distance, c.source_url, c.section_title,
              c.section_path, c.document_id, dict(c.metadata))
        for cid, c in plain.items()
    }
    install(monkeypatch, FakeCrossEncoder())

    results = service.retrieve(QUERY, top_k=3)

    for chunk in results:
        text, score, distance, url, title, path, doc_id, meta = before[chunk.chunk_id]
        assert (chunk.text, chunk.score, chunk.distance, chunk.source_url,
                chunk.section_title, chunk.section_path, chunk.document_id) == (
            text, score, distance, url, title, path, doc_id)
        extra = set(chunk.metadata) - set(meta)
        assert extra <= {"rerank_score"}
        assert {k: v for k, v in chunk.metadata.items() if k in meta} == meta
        assert chunk.metadata["source"] and chunk.metadata["page_number"] > 0
    assert results[0].metadata["rerank_score"] == 1.0


# 7 -------------------------------------------------------------------------
def test_empty_retrieval(monkeypatch):
    service = build_service(n=0)
    fake = install(monkeypatch, FakeCrossEncoder())

    assert service.retrieve(QUERY, top_k=3) == []
    assert fake.calls == []


# 8 -------------------------------------------------------------------------
def test_fewer_documents_than_top_n(monkeypatch):
    fake = install(monkeypatch, FakeCrossEncoder())

    two = build_service(n=2, mark_index=1).retrieve(QUERY, top_k=3)
    assert len(two) == 2 and two[0].text.endswith(TARGET_MARK)

    one = build_service(n=1).retrieve(QUERY, top_k=3)
    assert len(one) == 1
    assert len(fake.calls) == 1  # a single candidate is not sent to the model


def test_blank_passages_are_ranked_last(monkeypatch):
    chunks = [
        make_test_chunk(part=1, text="   ", document_id="blank"),
        make_test_chunk(part=2, text="real passage", document_id="real"),
    ]
    from rag.models import RetrievedChunk

    candidates = [
        RetrievedChunk(c.chunk_id, c.text, 0.5, 0.5, "", None, None, c.document_id, {})
        for c in chunks
    ]
    monkeypatch.setitem(
        retriever_module._CROSS_ENCODERS, settings.RERANK_MODEL,
        FakeCrossEncoder(scores=lambda n: [9.0, 0.1]),
    )

    out = rerank_chunks("q", candidates, 2)

    assert [c.document_id for c in out] == ["real", "blank"]


# 9 -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fake",
    [
        FakeCrossEncoder(boom=True),
        FakeCrossEncoder(scores=lambda n: [float("nan")] * n),
        FakeCrossEncoder(scores=lambda n: [1.0] * (n - 1)),
        None,  # model could not be loaded
    ],
    ids=["inference-error", "nan-scores", "wrong-length", "model-unavailable"],
)
def test_failure_falls_back_to_original_retrieval(monkeypatch, fake):
    service, _ = service_with_deep_target(monkeypatch)
    expected = baseline_order(service, monkeypatch, 3)
    install(monkeypatch, fake)

    results = service.retrieve(QUERY, top_k=3)

    assert [c.chunk_id for c in results] == expected
    assert all("rerank_score" not in c.metadata for c in results)
    assert "rerank_ms" not in service.last_latency


def test_disabled_matches_original_behaviour_and_skips_model(monkeypatch):
    service, _ = service_with_deep_target(monkeypatch)
    fake = install(monkeypatch, FakeCrossEncoder())
    expected = baseline_order(service, monkeypatch, 3)
    monkeypatch.setattr(settings, "RERANK_ENABLED", False)

    results = service.retrieve(QUERY, top_k=3)

    assert [c.chunk_id for c in results] == expected
    assert fake.calls == []


def test_failed_model_load_is_attempted_once(monkeypatch):
    attempts = []

    class BrokenCrossEncoder:
        def __init__(self, *args, **kwargs):
            attempts.append(args)
            raise OSError("model not reachable")

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.CrossEncoder = BrokenCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    assert retriever_module._get_cross_encoder("some/model") is None
    assert retriever_module._get_cross_encoder("some/model") is None
    assert len(attempts) == 1


def test_missing_sentence_transformers_does_not_break_retrieval(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import -> ImportError
    service, _ = service_with_deep_target(monkeypatch)
    expected = baseline_order(service, monkeypatch, 3)

    assert [c.chunk_id for c in service.retrieve(QUERY, top_k=3)] == expected


def test_latency_reports_rerank_stage(monkeypatch):
    service, _ = service_with_deep_target(monkeypatch)
    install(monkeypatch, FakeCrossEncoder())

    service.retrieve(QUERY, top_k=3)

    assert {"embed_ms", "search_ms", "total_ms", "rerank_ms"} <= set(service.last_latency)


# 10 ------------------------------------------------------------------------
class RecordingLLM(LLMService):
    model = "gen-stub"

    def __init__(self):
        self.prompts: list[str] = []

    def generate(self, system, prompt, max_tokens=1024, temperature=0.0):
        self.prompts.append(prompt)
        return LLMResponse("Answer from the first source [1].", LLMUsage(10, 5, self.model, 0.0))


class RelevantJudge(LLMService):
    model = "judge-stub"

    def generate(self, system, prompt, max_tokens=1024, temperature=0.0):
        payload = {"is_relevant": True, "confidence": 0.9, "reason": "ok", "missing": ""}
        return LLMResponse(json.dumps(payload), LLMUsage(1, 1, self.model, 0.0))


def assert_reranked_context_reached_llm(answer, llm, target_id):
    assert len(llm.prompts) == 1
    prompt = llm.prompts[0]
    first_block = prompt.split("[1]", 1)[1].split("[2]", 1)[0]
    assert TARGET_MARK in first_block  # re-ranked winner is source [1]
    assert answer.citations[0].chunk_id == target_id
    assert answer.retrieved[0].chunk_id == target_id
    assert len(answer.retrieved) == 3
    assert answer.has_sufficient_context is True


def test_linear_rag_pipeline_uses_reranked_chunks(monkeypatch):
    service, target_id = service_with_deep_target(monkeypatch)
    install(monkeypatch, FakeCrossEncoder())
    llm = RecordingLLM()

    answer = RagService(service, GenerationService(llm)).query("renewal topic filler")

    assert_reranked_context_reached_llm(answer, llm, target_id)


def test_agentic_pipeline_uses_reranked_chunks(monkeypatch):
    service, target_id = service_with_deep_target(monkeypatch)
    install(monkeypatch, FakeCrossEncoder())
    llm = RecordingLLM()
    judge = RelevantJudge()
    workflow = AgenticRagWorkflow(
        retriever=service,
        generator=GenerationService(llm),
        evaluator=LLMContextEvaluator(judge),
        rewriter=LLMQueryRewriter(judge),
    )

    answer = workflow.run("renewal topic filler")

    assert_reranked_context_reached_llm(answer, llm, target_id)
