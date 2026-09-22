"""Metric implementations checked against small, hand-computed examples.

No external LLM is called: the judge is exercised against a stub.
"""

import math

import pytest

from rag.evaluation.chunking_metrics import chunk_diagnostics
from rag.evaluation.dataset import EvalDataset, EvalQuestion
from rag.evaluation.rag_metrics import (
    LLMJudge,
    citation_completeness,
    citation_validity,
    deterministic_answer_metrics,
    insufficient_context_handling,
    lexical_groundedness,
    answer_token_f1,
)
from rag.evaluation.retrieval_metrics import (
    dcg_at_k,
    hit_rate_at_k,
    latency_summary,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.evaluation.runner import grade_results
from rag.llm.base import LLMResponse, LLMService
from rag.models import Citation, LLMUsage, RagAnswer, RetrievedChunk


# ---- retrieval metrics ----------------------------------------------------
def test_hit_rate():
    assert hit_rate_at_k([0, 1, 0], 1) == 0.0
    assert hit_rate_at_k([0, 1, 0], 2) == 1.0
    assert hit_rate_at_k([0, 0, 0], 3) == 0.0


def test_precision_denominator_is_k():
    assert precision_at_k([1, 1], 4) == 0.5  # only 2 returned, both relevant
    assert precision_at_k([1, 0, 1, 0], 4) == 0.5


def test_recall():
    assert recall_at_k([1, 0, 1], 3, total_relevant=2) == 1.0
    assert recall_at_k([1, 0, 0], 3, total_relevant=2) == 0.5
    assert recall_at_k([0], 1, total_relevant=0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([1, 0]) == 1.0
    assert reciprocal_rank([0, 0, 1]) == pytest.approx(1 / 3)
    assert reciprocal_rank([0, 0]) == 0.0


def test_dcg_matches_the_standard_worked_example():
    # Classic reference example: gains [3,2,3,0,1,2] -> DCG@6 = 6.861
    assert dcg_at_k([3, 2, 3, 0, 1, 2], 6) == pytest.approx(6.861, abs=1e-3)


def test_ndcg_perfect_and_demoted_ranking():
    assert ndcg_at_k([1, 0, 0], 3, [1, 0, 0]) == pytest.approx(1.0)
    # relevant item at rank 3 -> 1/log2(4) = 0.5
    assert ndcg_at_k([0, 0, 1], 3, [1, 0, 0]) == pytest.approx(0.5)
    assert ndcg_at_k([0, 0, 0], 3, [0, 0, 0]) == 0.0


def test_ndcg_uses_graded_relevance():
    assert ndcg_at_k([1, 3], 2, [3, 1]) < 1.0
    assert ndcg_at_k([3, 1], 2, [3, 1]) == pytest.approx(1.0)


def test_percentile_and_latency_summary():
    assert percentile([1, 2, 3, 4, 5], 100) == 5
    assert percentile([], 95) == 0.0
    summary = latency_summary([10.0, 20.0, 30.0])
    assert summary["mean_ms"] == 20.0 and summary["median_ms"] == 20.0
    assert latency_summary([])["count"] == 0


# ---- grading against ground truth ----------------------------------------
def _retrieved(url, path=None, doc="d1"):
    return RetrievedChunk(
        chunk_id="c", text="t", score=0.9, distance=0.1, source_url=url,
        section_title=(path or ["x"])[-1], section_path=path, document_id=doc,
    )


def test_grade_results_matches_url_case_and_slash_insensitively():
    question = EvalQuestion(
        id="q1", question="?",
        relevant_source_urls=["https://web.vodafone.com.eg/en/flex/"],
    )
    relevance, gains, flags = grade_results(
        question, [_retrieved("https://WEB.vodafone.com.eg/en/flex")]
    )
    assert relevance == [1.0] and flags["source_hit"] is True


def test_grade_results_matches_section_path():
    question = EvalQuestion(
        id="q1", question="?", relevant_sections=["Internet > Renewal"]
    )
    relevance, _, flags = grade_results(
        question, [_retrieved("https://x", path=["Internet", "Renewal"])]
    )
    assert relevance == [1.0] and flags["section_hit"] is True


def test_grade_results_uses_graded_relevance():
    question = EvalQuestion(
        id="q1", question="?",
        relevant_source_urls=["https://x"], graded_relevance={"https://x": 3.0},
    )
    _, gains, _ = grade_results(question, [_retrieved("https://x")])
    assert gains == [3.0]


# ---- RAG answer metrics ---------------------------------------------------
def _answer(text, n_chunks=2, citations=None, sufficient=True):
    retrieved = [
        RetrievedChunk(
            chunk_id=f"c{i}", text="You can renew your bundle via *880# for 70 EGP.",
            score=0.9, distance=0.1, source_url=f"https://x/{i}",
            section_title="Renewal", section_path=["Internet", "Renewal"], document_id="d1",
        )
        for i in range(n_chunks)
    ]
    return RagAnswer(
        query="How do I renew?", answer=text,
        citations=citations if citations is not None else [
            Citation(i + 1, f"https://x/{i}", "Renewal", ["Internet", "Renewal"], f"c{i}", "d1")
            for i in range(n_chunks)
        ],
        retrieved=retrieved, has_sufficient_context=sufficient,
    )


def test_citation_validity_detects_fabricated_index():
    assert citation_validity(_answer("Renew via *880# [1].")) == 1.0
    assert citation_validity(_answer("Renew via *880# [1][9].")) == 0.5
    assert citation_validity(_answer("Renew via *880#.")) is None  # undefined, not 0


def test_citation_completeness_counts_factual_sentences():
    # Both sentences carry numbers; only one is cited.
    score = citation_completeness(_answer("Costs 70 EGP [1]. Dial *880# to renew."))
    assert score == pytest.approx(0.5)
    # No factual sentence -> metric unavailable, not zero.
    assert citation_completeness(_answer("Thank you for asking.")) is None


def test_lexical_groundedness_flags_invented_content():
    grounded = lexical_groundedness(_answer("You can renew your bundle via *880#."))
    invented = lexical_groundedness(
        _answer("Purchase quantum satellite roaming packages in Antarctica today.")
    )
    assert grounded > invented


def test_answer_token_f1_requires_a_reference():
    assert answer_token_f1("renew your bundle", None) is None
    assert answer_token_f1("renew your bundle", "renew your bundle") == pytest.approx(1.0)
    assert answer_token_f1("totally different words", "renew your bundle") == 0.0


def test_insufficient_context_handling_scores_refusal():
    refused = _answer("no info", sufficient=False)
    answered = _answer("some answer", sufficient=True)
    assert insufficient_context_handling(refused, expected_insufficient=True) == 1.0
    assert insufficient_context_handling(answered, expected_insufficient=True) == 0.0
    assert insufficient_context_handling(answered, expected_insufficient=None) is None


def test_deterministic_metrics_report_none_when_unavailable():
    metrics = deterministic_answer_metrics(_answer("Renew via *880# [1]."))
    assert metrics["answer_token_f1"] is None
    assert metrics["insufficient_context_handling"] is None
    assert metrics["citation_validity"] == 1.0


# ---- LLM judge (stubbed) --------------------------------------------------
class StubJudgeLLM(LLMService):
    model = "stub-judge"

    def __init__(self, payload):
        self.payload = payload

    def generate(self, system, prompt, max_tokens=1024, temperature=0.0):
        return LLMResponse(self.payload, LLMUsage(100, 50, self.model, 0.002))


def test_judge_parses_scores_and_tracks_usage():
    judge = LLMJudge(StubJudgeLLM(
        '{"faithfulness":0.9,"answer_relevance":1.0,"context_relevance":0.8,'
        '"context_precision":0.7,"context_recall":0.6,"answer_correctness":null,'
        '"unsupported_claim_rate":0.1,"reason":"ok"}'
    ))
    result = judge.score("q", "a", [], None)
    assert result.scores["faithfulness"] == 0.9
    assert result.scores["answer_correctness"] is None
    assert judge.usage_input == 100 and judge.usage_output == 50


def test_judge_handles_unparseable_output():
    result = LLMJudge(StubJudgeLLM("not json at all")).score("q", "a", [])
    assert result.error == "unparseable_judge_output"
    assert all(v is None for v in result.scores.values())


def test_judge_clamps_out_of_range_scores():
    judge = LLMJudge(StubJudgeLLM('{"faithfulness": 5, "answer_relevance": -2}'))
    scores = judge.score("q", "a", []).scores
    assert scores["faithfulness"] == 1.0 and scores["answer_relevance"] == 0.0


# ---- chunking diagnostics -------------------------------------------------
def _chunk(i, text, section_index=0):
    from tests.factories import make_test_chunk

    return make_test_chunk(part=i + 1, text=text, section_index=section_index,
                           unit_count=2)


def test_chunk_diagnostics_detects_duplicates_and_sizes():
    chunks = [_chunk(0, "a" * 300), _chunk(1, "a" * 300), _chunk(2, "b" * 100)]
    report = chunk_diagnostics(chunks, document_count=1, section_count=1, min_size=200)
    assert report["total_chunks"] == 3
    assert report["duplicate_chunks"] == 1
    assert report["size"]["max"] == 300
    assert report["pct_below_min_size"] == pytest.approx(33.33, abs=0.01)
    assert report["section_boundary_violations"] == 0


def test_chunk_diagnostics_handles_empty_input():
    assert chunk_diagnostics([], 0, 0)["total_chunks"] == 0


# ---- dataset schema -------------------------------------------------------
def test_question_knows_when_it_lacks_ground_truth():
    assert not EvalQuestion(id="q", question="?").has_retrieval_ground_truth()
    assert EvalQuestion(
        id="q", question="?", relevant_source_urls=["https://x"]
    ).has_retrieval_ground_truth()


def test_dataset_roundtrip(tmp_path):
    dataset = EvalDataset(questions=[EvalQuestion(id="q1", question="ازاي أجدد؟")])
    path = tmp_path / "eval.json"
    dataset.save(path)
    assert EvalDataset.load(path).questions[0].question == "ازاي أجدد؟"


def test_missing_dataset_raises_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="build-dataset"):
        EvalDataset.load(tmp_path / "nope.json")
