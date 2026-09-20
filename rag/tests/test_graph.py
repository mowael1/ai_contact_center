"""Agentic LangGraph workflow: branch, loop, give-up, and the separate judge.

No real LLM is called - the judge and generator are scripted stubs.
"""

import json

import pytest

from rag.embeddings.local_providers import HashEmbedding
from rag.graph.workflow import AgenticRagWorkflow
from rag.llm.base import LLMResponse, LLMService, StreamChunk
from rag.models import Chunk, LLMUsage
from rag.services.evaluator import LLMContextEvaluator, LLMQueryRewriter
from rag.services.generator import GenerationService
from rag.services.retriever import RetrievalService
from rag.vectorstore.memory_store import InMemoryVectorStore

REWRITER_MARKER = "rewrite failed search queries"


def verdict(relevant: bool, confidence: float = 0.9, missing: str = "") -> str:
    return json.dumps(
        {
            "is_relevant": relevant,
            "confidence": confidence,
            "reason": "scripted",
            "missing": missing,
        }
    )


class ScriptedJudge(LLMService):
    """Returns queued verdicts; answers rewrite prompts with a fixed query."""

    model = "judge-stub"

    def __init__(self, verdicts=(), rewritten="rewritten flex query"):
        self.verdicts = list(verdicts)
        self.rewritten = rewritten
        self.verdict_calls = 0
        self.rewrite_calls = 0

    def generate(self, system, prompt, max_tokens=1024, temperature=0.0):
        if REWRITER_MARKER in system:
            self.rewrite_calls += 1
            return LLMResponse(self.rewritten, LLMUsage(10, 5, self.model, 0.0))
        self.verdict_calls += 1
        payload = self.verdicts.pop(0) if self.verdicts else verdict(True)
        return LLMResponse(payload, LLMUsage(20, 10, self.model, 0.0))


class GeneratorStub(LLMService):
    model = "gen-stub"

    def __init__(self, text="You renew via *880# [1]."):
        self.text = text
        self.calls = 0
        self.stream_calls = 0

    def generate(self, system, prompt, max_tokens=1024, temperature=0.0):
        self.calls += 1
        return LLMResponse(self.text, LLMUsage(100, 20, self.model, 0.001))

    def generate_stream(self, system, prompt, max_tokens=1024, temperature=0.0):
        self.stream_calls += 1
        accumulated = ""
        for word in self.text.split(" "):
            accumulated += (" " if accumulated else "") + word
            yield StreamChunk(
                delta=(" " if accumulated != word else "") + word, text=accumulated
            )
        yield StreamChunk(
            text=accumulated, done=True, usage=LLMUsage(100, 20, self.model, 0.001)
        )


def make_workflow(judge, generator_llm=None, max_attempts=3):
    embeddings = HashEmbedding()
    store = InMemoryVectorStore()
    chunk = Chunk(
        chunk_id="c1", document_id="d1", source_url="https://web.vodafone.com.eg/en/flex",
        text="You can renew your bundle by dialling *880#.", section_title="Renewal",
        section_path=["Internet", "Renewal"], heading_level=3, section_index=0,
        chunk_index=0, chunking_method="semantic",
        section_extraction_method="html_structure", language="en", domain="x",
    )
    store.upsert_chunks([chunk], embeddings.embed_documents([chunk.text]))
    return AgenticRagWorkflow(
        retriever=RetrievalService(store, embeddings),
        generator=GenerationService(generator_llm or GeneratorStub()),
        evaluator=LLMContextEvaluator(judge),
        rewriter=LLMQueryRewriter(judge),
        max_attempts=max_attempts,
    )


# ---- branching ------------------------------------------------------------
def test_relevant_context_goes_straight_to_generate():
    judge = ScriptedJudge([verdict(True)])
    answer = make_workflow(judge).run("How do I renew?")
    assert answer.has_sufficient_context is True
    assert len(answer.attempts) == 1
    assert judge.rewrite_calls == 0
    assert any(t.startswith("generate") for t in answer.trace)


def test_irrelevant_then_relevant_loops_once():
    judge = ScriptedJudge([verdict(False, missing="the USSD code"), verdict(True)])
    answer = make_workflow(judge).run("How do I renew?")
    assert len(answer.attempts) == 2
    assert judge.rewrite_calls == 1
    assert answer.has_sufficient_context is True
    assert answer.attempts[0]["is_relevant"] is False
    assert answer.attempts[1]["is_relevant"] is True


def test_never_relevant_gives_up_without_citations():
    judge = ScriptedJudge([verdict(False)] * 5)
    answer = make_workflow(judge, max_attempts=3).run("ما هو سعر السهم؟")
    assert answer.has_sufficient_context is False
    assert answer.citations == []
    assert len(answer.attempts) == 3
    assert any("give_up" in t for t in answer.trace)


def test_give_up_answers_in_the_question_language():
    judge = ScriptedJudge([verdict(False)] * 5)
    arabic = make_workflow(judge, max_attempts=1).run("ازاي أجدد باقتي؟")
    judge2 = ScriptedJudge([verdict(False)] * 5)
    english = make_workflow(judge2, max_attempts=1).run("How do I renew?")
    assert any("؀" <= c <= "ۿ" for c in arabic.answer)
    assert "knowledge base" in english.answer.lower()


def test_max_attempts_is_respected():
    judge = ScriptedJudge([verdict(False)] * 10)
    answer = make_workflow(judge, max_attempts=2).run("q")
    assert len(answer.attempts) == 2
    assert judge.rewrite_calls == 1  # one rewrite between two attempts


def test_generator_is_not_called_when_giving_up():
    generator = GeneratorStub()
    judge = ScriptedJudge([verdict(False)] * 5)
    make_workflow(judge, generator, max_attempts=2).run("q")
    assert generator.calls == 0


def test_rewritten_query_is_actually_used():
    judge = ScriptedJudge(
        [verdict(False), verdict(True)], rewritten="Flex bundle renewal *880#"
    )
    answer = make_workflow(judge).run("How do I renew?")
    assert answer.attempts[1]["query"] == "Flex bundle renewal *880#"


# ---- the judge is separate -----------------------------------------------
def test_judge_and_generator_are_different_models():
    judge = ScriptedJudge([verdict(True)])
    generator = GeneratorStub()
    workflow = make_workflow(judge, generator)
    workflow.run("q")
    assert judge.verdict_calls == 1
    assert generator.calls == 1
    assert workflow.evaluator.llm.model != workflow.generator.llm.model


def test_low_confidence_relevance_is_rejected():
    """A 'relevant' verdict below the threshold must not pass."""
    judge = ScriptedJudge([verdict(True, confidence=0.2), verdict(True, confidence=0.95)])
    workflow = make_workflow(judge)
    workflow.evaluator.threshold = 0.5
    answer = workflow.run("q")
    assert answer.attempts[0]["is_relevant"] is False
    assert len(answer.attempts) == 2


def test_evaluator_fails_open_when_the_judge_errors():
    class BrokenJudge(LLMService):
        model = "broken"

        def generate(self, *a, **k):
            raise RuntimeError("judge down")

    evaluator = LLMContextEvaluator(BrokenJudge())
    chunk_verdict = evaluator.evaluate("q", [object()])
    assert chunk_verdict.is_relevant is True
    assert "unavailable" in chunk_verdict.reason


def test_evaluator_returns_irrelevant_for_empty_context():
    evaluator = LLMContextEvaluator(ScriptedJudge())
    assert evaluator.evaluate("q", []).is_relevant is False


def test_unparseable_judge_output_fails_open():
    class Garbage(LLMService):
        model = "g"

        def generate(self, *a, **k):
            return LLMResponse("not json", LLMUsage(1, 1, "g", 0.0))

    assert LLMContextEvaluator(Garbage()).evaluate("q", [object()]).is_relevant is True


def test_rewriter_falls_back_to_the_original_query_on_failure():
    class Broken(LLMService):
        model = "b"

        def generate(self, *a, **k):
            raise RuntimeError("down")

    assert LLMQueryRewriter(Broken()).rewrite("original query", [], 1) == "original query"


def test_rewriter_rejects_an_empty_rewrite():
    judge = ScriptedJudge(rewritten="  ")
    assert LLMQueryRewriter(judge).rewrite("original query", [], 1) == "original query"


# ---- streaming through the graph -----------------------------------------
def test_run_stream_emits_deltas_then_the_final_answer():
    judge = ScriptedJudge([verdict(True)])
    generator = GeneratorStub()
    workflow = make_workflow(judge, generator)

    rebuilt = ""
    final = None
    for piece in workflow.run_stream("How do I renew?"):
        if piece.done:
            final = piece.answer
        else:
            if piece.replaces_from is not None:
                rebuilt = rebuilt[: piece.replaces_from]
            rebuilt += piece.delta
    assert final is not None
    assert rebuilt == final.answer
    assert final.citations


def test_run_stream_does_not_generate_twice():
    judge = ScriptedJudge([verdict(True)])
    generator = GeneratorStub()
    for _ in make_workflow(judge, generator).run_stream("q"):
        pass
    assert generator.calls == 0       # blocking path untouched
    assert generator.stream_calls == 1


def test_run_stream_carries_the_loop_audit_trail():
    judge = ScriptedJudge([verdict(False), verdict(True)])
    final = None
    for piece in make_workflow(judge).run_stream("q"):
        if piece.done:
            final = piece.answer
    assert len(final.attempts) == 2
    assert any("rewrite" in t for t in final.trace)


def test_run_stream_gives_up_without_calling_the_generator():
    judge = ScriptedJudge([verdict(False)] * 5)
    generator = GeneratorStub()
    final = None
    for piece in make_workflow(judge, generator, max_attempts=2).run_stream("q"):
        if piece.done:
            final = piece.answer
    assert final.has_sufficient_context is False
    assert final.citations == []
    assert generator.calls == 0 and generator.stream_calls == 0
