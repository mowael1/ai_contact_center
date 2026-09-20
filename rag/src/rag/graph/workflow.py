"""LangGraph agentic retrieval workflow.

    Question
       │
       ▼
    retrieve ──► evaluate_context ──► relevant?
       ▲                                │
       │                         ┌──────┴──────┐
       │                        YES            NO
       │                         │              │
       │                         ▼              ▼
       │                     generate      attempts left?
       │                                    │         │
       └──────── rewrite_query ◄───── YES   │        NO
                                            │         │
                                            └────► give_up

Why a graph here and not plain Python: this workflow has **state** (attempt
counter, query history, per-attempt verdicts), **branching** (the relevance
decision) and a **loop** (rewrite and retry). Those are exactly the conditions
under which LangGraph earns its keep - unlike the original linear pipeline,
which is still available via ``AGENTIC_ENABLED=false``.

Only ``langgraph`` itself is used. LangChain's abstractions are not: it arrives
only as a transitive dependency of langgraph.
"""

from __future__ import annotations

import time
from typing import Any, Iterator, Optional

from langgraph.graph import END, START, StateGraph

from rag.config import settings
from rag.graph.state import RagState
from rag.logging_utils import get_logger
from rag.models import RagAnswer
from rag.services.evaluator import LLMContextEvaluator, LLMQueryRewriter
from rag.services.generator import AnswerChunk, GenerationService
from rag.services.prompts import NO_CONTEXT_ANSWERS
from rag.services.retriever import RetrievalService
from rag.text_utils import detect_language

logger = get_logger(__name__)


class AgenticRagWorkflow:
    """Builds and runs the retrieve/evaluate/rewrite/generate graph."""

    def __init__(
        self,
        retriever: RetrievalService,
        generator: GenerationService,
        evaluator: LLMContextEvaluator,
        rewriter: LLMQueryRewriter,
        max_attempts: Optional[int] = None,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.evaluator = evaluator
        self.rewriter = rewriter
        self.max_attempts = max_attempts or settings.MAX_RETRIEVAL_ATTEMPTS
        self.graph = self._build()

    # -- graph construction -------------------------------------------------
    def _build(self):
        builder = StateGraph(RagState)
        builder.add_node("retrieve", self._retrieve)
        builder.add_node("evaluate_context", self._evaluate)
        builder.add_node("rewrite_query", self._rewrite)
        builder.add_node("generate", self._generate)
        builder.add_node("give_up", self._give_up)

        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "evaluate_context")
        builder.add_conditional_edges(
            "evaluate_context",
            self._route,
            {"generate": "generate", "rewrite": "rewrite_query", "give_up": "give_up"},
        )
        builder.add_edge("rewrite_query", "retrieve")  # the loop
        builder.add_edge("generate", END)
        builder.add_edge("give_up", END)
        return builder.compile()

    # -- nodes --------------------------------------------------------------
    def _retrieve(self, state: RagState) -> dict[str, Any]:
        query = state.get("query") or state["question"]
        attempt = state.get("attempt", 0) + 1
        started = time.perf_counter()
        chunks = self.retriever.retrieve(
            query, top_k=state.get("top_k") or settings.TOP_K,
            filters=state.get("filters"),
        )
        elapsed = (time.perf_counter() - started) * 1000
        logger.info("[graph] attempt %d retrieved %d chunks for %r",
                    attempt, len(chunks), query[:60])
        return {
            "query": query,
            "attempt": attempt,
            "chunks": chunks,
            "trace": [f"retrieve#{attempt}: {len(chunks)} chunks for {query!r}"],
            "latency_ms": {
                **(state.get("latency_ms") or {}),
                f"retrieve_{attempt}_ms": round(elapsed, 3),
            },
        }

    def _evaluate(self, state: RagState) -> dict[str, Any]:
        chunks = state.get("chunks") or []
        started = time.perf_counter()
        verdict = self.evaluator.evaluate(state["question"], chunks)
        elapsed = (time.perf_counter() - started) * 1000
        attempt = state.get("attempt", 1)
        logger.info("[graph] attempt %d verdict relevant=%s confidence=%.2f (%s)",
                    attempt, verdict.is_relevant, verdict.confidence, verdict.reason[:80])
        return {
            "is_relevant": verdict.is_relevant,
            "reason": verdict.reason,
            "missing": verdict.missing,
            "confidence": verdict.confidence,
            "attempts": [{
                "attempt": attempt,
                "query": state.get("query", ""),
                "retrieved": len(chunks),
                "top_score": round(chunks[0].score, 4) if chunks else 0.0,
                "is_relevant": verdict.is_relevant,
                "confidence": verdict.confidence,
                "reason": verdict.reason,
                "missing": verdict.missing,
                "latency_ms": round(elapsed, 3),
            }],
            "trace": [
                f"evaluate#{attempt}: relevant={verdict.is_relevant} "
                f"confidence={verdict.confidence:.2f}"
            ],
            "latency_ms": {
                **(state.get("latency_ms") or {}),
                f"evaluate_{attempt}_ms": round(elapsed, 3),
            },
        }

    def _rewrite(self, state: RagState) -> dict[str, Any]:
        attempt = state.get("attempt", 1)
        new_query = self.rewriter.rewrite(
            query=state.get("query", state["question"]),
            chunks=state.get("chunks") or [],
            attempt=attempt,
            original_question=state["question"],
            missing=state.get("missing", ""),
        )
        return {
            "query": new_query,
            "trace": [f"rewrite#{attempt}: -> {new_query!r}"],
        }

    def _generate(self, state: RagState) -> dict[str, Any]:
        answer = self.generator.generate(state["question"], state.get("chunks") or [])
        return {
            "answer": answer.answer,
            "citations": answer.citations,
            "has_sufficient_context": answer.has_sufficient_context,
            "usage": answer.usage,
            "trace": [f"generate: {len(answer.citations)} citations"],
            "latency_ms": {
                **(state.get("latency_ms") or {}),
                **answer.latency_ms,
            },
        }

    def _give_up(self, state: RagState) -> dict[str, Any]:
        """Every attempt failed the relevance check - refuse rather than guess."""
        language = detect_language(state["question"])
        logger.info("[graph] giving up after %d attempts", state.get("attempt", 0))
        return {
            "answer": NO_CONTEXT_ANSWERS.get(language, NO_CONTEXT_ANSWERS["en"]),
            "citations": [],
            "has_sufficient_context": False,
            "trace": [f"give_up after {state.get('attempt', 0)} attempts"],
        }

    # -- routing ------------------------------------------------------------
    def _route(self, state: RagState) -> str:
        if state.get("is_relevant"):
            return "generate"
        max_attempts = state.get("max_attempts") or self.max_attempts
        if state.get("attempt", 1) >= max_attempts:
            return "give_up"
        return "rewrite"

    def run_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        max_attempts: Optional[int] = None,
    ) -> Iterator[AnswerChunk]:
        """Run the loop, then stream the final generation.

        The retrieve/evaluate/rewrite cycle is not streamed - it produces
        verdicts, not prose. Only the answer itself streams, which is what a
        caller actually renders. The loop's audit trail rides on the final
        event's ``answer.attempts`` / ``answer.trace``.
        """
        started = time.perf_counter()
        state: RagState = {
            "question": question,
            "query": question,
            "top_k": top_k or settings.TOP_K,
            "filters": filters,
            "attempt": 0,
            "max_attempts": max_attempts or self.max_attempts,
            "attempts": [],
            "trace": [],
            "latency_ms": {},
        }
        # Stop before generation so the answer can be streamed by this method.
        resolved = self.graph.invoke(state, interrupt_before=["generate", "give_up"])

        chunks = resolved.get("chunks") or []
        attempts = resolved.get("attempts") or []
        trace = list(resolved.get("trace") or [])
        latency = dict(resolved.get("latency_ms") or {})

        if not resolved.get("is_relevant"):
            final = self._give_up(resolved)
            answer = RagAnswer(
                query=question, answer=final["answer"], citations=[], retrieved=chunks,
                has_sufficient_context=False,
                latency_ms={**latency, "total_ms": round((time.perf_counter() - started) * 1000, 3)},
                attempts=attempts, trace=trace + final["trace"],
            )
            yield AnswerChunk(delta=answer.answer, text=answer.answer)
            yield AnswerChunk(text=answer.answer, done=True, answer=answer)
            return

        for piece in self.generator.generate_stream(question, chunks):
            if piece.done and piece.answer is not None:
                piece.answer.attempts = attempts
                piece.answer.trace = trace + [f"generate: {len(piece.answer.citations)} citations"]
                piece.answer.latency_ms = {
                    **latency, **piece.answer.latency_ms,
                    "total_ms": round((time.perf_counter() - started) * 1000, 3),
                }
            yield piece

    # -- entry point --------------------------------------------------------
    def run(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[dict[str, Any]] = None,
        max_attempts: Optional[int] = None,
    ) -> RagAnswer:
        started = time.perf_counter()
        initial: RagState = {
            "question": question,
            "query": question,
            "top_k": top_k or settings.TOP_K,
            "filters": filters,
            "attempt": 0,
            "max_attempts": max_attempts or self.max_attempts,
            "attempts": [],
            "trace": [],
            "latency_ms": {},
        }
        final = self.graph.invoke(initial)

        latency = dict(final.get("latency_ms") or {})
        latency["total_ms"] = round((time.perf_counter() - started) * 1000, 3)

        answer = RagAnswer(
            query=question,
            answer=final.get("answer", ""),
            citations=final.get("citations") or [],
            retrieved=final.get("chunks") or [],
            has_sufficient_context=bool(final.get("has_sufficient_context")),
            usage=final.get("usage"),
            latency_ms=latency,
            attempts=final.get("attempts") or [],
            trace=final.get("trace") or [],
        )
        return answer
