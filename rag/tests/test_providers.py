"""Gemini and EmbeddingGemma providers, exercised against mocked HTTP."""

import json

import httpx
import pytest

from rag.embeddings.gemma_provider import (
    DOCUMENT_PREFIX,
    QUERY_PREFIX,
    HFInferenceEmbeddingGemma,
    format_document,
    format_query,
)
from rag.llm.gemini_provider import GeminiLLM, gemini_cost


def gemini_body(text, prompt_tokens=100, output_tokens=20):
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": output_tokens,
        },
    }


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# ---- Gemini ---------------------------------------------------------------
def test_generate_returns_text_and_usage():
    def handler(request):
        assert "generateContent" in str(request.url)
        return httpx.Response(200, json=gemini_body("تقدر تجدد من *880# [1]."))

    llm = GeminiLLM(model="gemini-2.5-pro", api_key="test-key")
    llm._client = mock_client(handler)
    response = llm.generate("system", "prompt")
    assert response.text == "تقدر تجدد من *880# [1]."
    assert response.usage.input_tokens == 100
    assert response.usage.output_tokens == 20
    assert response.usage.estimated_cost_usd > 0


def test_api_key_travels_in_a_header_not_the_url():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["header"] = request.headers.get("x-goog-api-key")
        return httpx.Response(200, json=gemini_body("ok"))

    llm = GeminiLLM(model="gemini-2.5-flash", api_key="super-secret-key")
    llm._client = mock_client(handler)
    llm.generate("s", "p")
    assert seen["header"] == "super-secret-key"
    assert "super-secret-key" not in seen["url"]


def test_thinking_budget_is_disabled_for_grounded_extraction():
    captured = {}

    def handler(request):
        captured.update(json.loads(request.content))
        return httpx.Response(200, json=gemini_body("ok"))

    llm = GeminiLLM(model="gemini-2.5-pro", api_key="k")
    llm._client = mock_client(handler)
    llm.generate("s", "p", max_tokens=512, temperature=0.0)
    config = captured["generationConfig"]
    assert config["thinkingConfig"]["thinkingBudget"] == 0
    assert config["maxOutputTokens"] == 512
    assert captured["systemInstruction"]["parts"][0]["text"] == "s"


def test_thought_tokens_count_towards_output():
    def handler(request):
        body = gemini_body("x", output_tokens=10)
        body["usageMetadata"]["thoughtsTokenCount"] = 40
        return httpx.Response(200, json=body)

    llm = GeminiLLM(model="gemini-2.5-pro", api_key="k")
    llm._client = mock_client(handler)
    assert llm.generate("s", "p").usage.output_tokens == 50


def test_empty_candidates_yield_empty_text():
    llm = GeminiLLM(model="gemini-2.5-flash", api_key="k")
    llm._client = mock_client(lambda r: httpx.Response(200, json={"candidates": []}))
    assert llm.generate("s", "p").text == ""


def test_missing_api_key_raises_a_clear_error(monkeypatch):
    from rag.config import settings

    monkeypatch.setattr(settings, "GOOGLE_API_KEY", "")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        GeminiLLM(model="gemini-2.5-pro")





@pytest.mark.parametrize(
    "model,expected_nonzero",
    [("gemini-2.5-pro", True), ("gemini-2.5-flash", True), ("unknown-model", False)],
)
def test_pricing_table(model, expected_nonzero):
    cost = gemini_cost(model, 1_000_000, 1_000_000)
    assert (cost is not None and cost > 0) is expected_nonzero


# ---- EmbeddingGemma -------------------------------------------------------
def test_task_prefixes_match_the_model_card():
    assert format_query("renew") == f"{QUERY_PREFIX}renew"
    assert format_document("body") == f"{DOCUMENT_PREFIX}body"
    assert format_document("body", title="Renewal").startswith("title: Renewal | text:")


def test_query_and_document_prefixes_are_applied():
    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        count = len(sent["inputs"])
        return httpx.Response(200, json=[[0.1] * 768 for _ in range(count)])

    service = HFInferenceEmbeddingGemma(api_key="hf_test")
    service._client = mock_client(handler)

    service.embed_documents(["first", "second"])
    assert all(i.startswith(DOCUMENT_PREFIX) for i in sent["inputs"])

    service.embed_query("how do I renew?")
    assert sent["inputs"][0].startswith(QUERY_PREFIX)


def test_dimension_is_768():
    def handler(request):
        return httpx.Response(200, json=[[0.1] * 768])

    service = HFInferenceEmbeddingGemma(api_key="hf_test")
    service._client = mock_client(handler)
    assert len(service.embed_query("q")) == 768
    assert service.dimension == 768


def test_token_pooled_response_is_mean_pooled():
    def handler(request):
        # [batch][tokens][dim] - not pre-pooled
        return httpx.Response(200, json=[[[0.0, 2.0], [2.0, 0.0]]])

    service = HFInferenceEmbeddingGemma(api_key="hf_test")
    service._client = mock_client(handler)
    assert service.embed_query("q") == [1.0, 1.0]


def test_batching_splits_large_inputs():
    batches = []

    def handler(request):
        inputs = json.loads(request.content)["inputs"]
        batches.append(len(inputs))
        return httpx.Response(200, json=[[0.1] * 768 for _ in inputs])

    service = HFInferenceEmbeddingGemma(api_key="hf_test", batch_size=4)
    service._client = mock_client(handler)
    vectors = service.embed_documents([f"text {i}" for i in range(10)])
    assert len(vectors) == 10
    assert batches == [4, 4, 2]


def test_gated_model_rejection_explains_the_licence():
    def handler(request):
        return httpx.Response(403, json={"error": "forbidden"})

    service = HFInferenceEmbeddingGemma(api_key="hf_test")
    service._client = mock_client(handler)
    with pytest.raises(RuntimeError, match="gated"):
        service.embed_query("q")


def test_missing_hf_token_raises_a_clear_error(monkeypatch):
    from rag.config import settings

    monkeypatch.setattr(settings, "HF_API_TOKEN", "")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_API_TOKEN"):
        HFInferenceEmbeddingGemma()


def test_model_info_reports_the_backend():
    info = HFInferenceEmbeddingGemma(api_key="hf_test").model_info()
    assert info["model"] == "google/embeddinggemma-300m"
    assert info["backend"] == "huggingface_inference_api"
    assert "hf_test" not in str(info)


# ---- generic HF Inference provider ----------------------------------------
def test_e5_models_get_query_and_passage_prefixes():
    from rag.embeddings.gemma_provider import HFInferenceEmbedding

    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=[[0.1] * 1024 for _ in sent["inputs"]])

    service = HFInferenceEmbedding("intfloat/multilingual-e5-large", api_key="hf_test")
    service._client = mock_client(handler)

    service.embed_documents(["bundle renewal"])
    assert sent["inputs"][0] == "passage: bundle renewal"
    service.embed_query("how do I renew?")
    assert sent["inputs"][0] == "query: how do I renew?"


def test_bge_m3_gets_no_prefix():
    from rag.embeddings.gemma_provider import HFInferenceEmbedding

    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=[[0.1] * 1024])

    service = HFInferenceEmbedding("BAAI/bge-m3", api_key="hf_test")
    service._client = mock_client(handler)
    service.embed_query("ازاي أجدد باقتي؟")
    assert sent["inputs"][0] == "ازاي أجدد باقتي؟"


def test_generic_provider_discovers_dimension_from_the_response():
    from rag.embeddings.gemma_provider import HFInferenceEmbedding

    service = HFInferenceEmbedding("BAAI/bge-m3", api_key="hf_test")
    service._client = mock_client(lambda r: httpx.Response(200, json=[[0.1] * 1024]))
    assert len(service.embed_query("x")) == 1024
    assert service.dimension == 1024


def test_embeddinggemma_prefixes_survive_the_generic_path():
    from rag.embeddings.gemma_provider import HFInferenceEmbedding

    sent = {}

    def handler(request):
        sent.update(json.loads(request.content))
        return httpx.Response(200, json=[[0.1] * 768])

    service = HFInferenceEmbedding("google/embeddinggemma-300m", api_key="hf_test")
    service._client = mock_client(handler)
    service.embed_query("renew")
    assert sent["inputs"][0].startswith(QUERY_PREFIX)


def test_factory_builds_the_generic_provider():
    from rag.embeddings.factory import build_embedding_service

    service = build_embedding_service(
        provider="hf_inference", model="BAAI/bge-m3", cache=False
    )
    assert service.model == "BAAI/bge-m3"
    assert service.model_info()["provider"] == "HFInferenceEmbedding"




