"""Evaluation orchestration, split by pipeline stage.

Stages, each runnable on its own:

1. chunking   - structural diagnostics (no vector store, no LLM)
2. retrieval  - HitRate / Recall / Precision / MRR / nDCG / latency
3. rag        - answer + citation quality (deterministic + optional judge)

A metric with no ground truth in the dataset is reported as ``null`` under
``unavailable_metrics``, never fabricated.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from rag.config import settings
from rag.evaluation.dataset import EvalDataset, EvalQuestion
from rag.evaluation.rag_metrics import (
    JUDGE_KEYS,
    LLMJudge,
    deterministic_answer_metrics,
)
from rag.evaluation.retrieval_metrics import (
    aggregate,
    hit_rate_at_k,
    latency_summary,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag.logging_utils import get_logger
from rag.models import RetrievedChunk
from rag.services.rag_service import RagService
from rag.services.retriever import RetrievalService

logger = get_logger(__name__)

K_VALUES = (1, 3, 5, 10)
NDCG_K = (3, 5, 10)


def _normalize_url(url: str) -> str:
    return (url or "").strip().rstrip("/").lower()


def _section_key(path: Optional[Sequence[str]], title: Optional[str]) -> str:
    if path:
        return " > ".join(path).strip().lower()
    return (title or "").strip().lower()


def grade_results(
    question: EvalQuestion, results: Sequence[RetrievedChunk]
) -> tuple[list[float], list[float], dict[str, bool]]:
    """Score each retrieved chunk against the question's ground truth.

    Returns (binary relevance, graded gains, per-dimension hit flags).
    """
    urls = {_normalize_url(u) for u in question.relevant_source_urls}
    doc_ids = set(question.relevant_document_ids)
    sections = {s.strip().lower() for s in question.relevant_sections}
    graded = {_normalize_url(k): v for k, v in question.graded_relevance.items()}

    relevance: list[float] = []
    gains: list[float] = []
    for chunk in results:
        url = _normalize_url(chunk.source_url)
        section = _section_key(chunk.section_path, chunk.section_title)
        is_relevant = (
            (bool(urls) and url in urls)
            or (bool(doc_ids) and chunk.document_id in doc_ids)
            or (bool(sections) and section in sections)
        )
        relevance.append(1.0 if is_relevant else 0.0)
        gains.append(graded.get(url, 1.0 if is_relevant else 0.0))

    flags = {
        "source_hit": any(
            _normalize_url(c.source_url) in urls for c in results
        ) if urls else False,
        "section_hit": any(
            _section_key(c.section_path, c.section_title) in sections for c in results
        ) if sections else False,
    }
    return relevance, gains, flags


@dataclass
class RetrievalEvalResult:
    per_query: list[dict[str, Any]]
    summary: dict[str, Any]


def evaluate_retrieval(
    retriever: RetrievalService,
    dataset: EvalDataset,
    top_k: int = 10,
) -> RetrievalEvalResult:
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    total_latency: list[float] = []
    embed_latency: list[float] = []
    search_latency: list[float] = []
    section_hits: dict[int, list[float]] = {k: [] for k in K_VALUES}
    source_hits: dict[int, list[float]] = {k: [] for k in K_VALUES}
    skipped: list[str] = []

    for question in dataset.questions:
        if not question.has_retrieval_ground_truth():
            skipped.append(question.id)
            continue

        started = time.perf_counter()
        results = retriever.retrieve(question.question, top_k=top_k)
        elapsed = (time.perf_counter() - started) * 1000
        total_latency.append(elapsed)
        embed_latency.append(retriever.last_latency.get("embed_ms", 0.0))
        search_latency.append(retriever.last_latency.get("search_ms", 0.0))

        relevance, gains, _ = grade_results(question, results)
        total_relevant = max(
            1,
            len(question.relevant_source_urls)
            or len(question.relevant_document_ids)
            or len(question.relevant_sections),
        )

        metrics: dict[str, float] = {}
        for k in K_VALUES:
            metrics[f"HitRate@{k}"] = hit_rate_at_k(relevance, k)
            metrics[f"Recall@{k}"] = recall_at_k(relevance, k, total_relevant)
            metrics[f"Precision@{k}"] = precision_at_k(relevance, k)
        metrics["MRR"] = reciprocal_rank(relevance)
        for k in NDCG_K:
            metrics[f"nDCG@{k}"] = ndcg_at_k(gains, k)

        # Per-dimension accuracy at each cut-off.
        urls = {_normalize_url(u) for u in question.relevant_source_urls}
        sections = {s.strip().lower() for s in question.relevant_sections}
        for k in K_VALUES:
            top = results[:k]
            if urls:
                source_hits[k].append(
                    1.0 if any(_normalize_url(c.source_url) in urls for c in top) else 0.0
                )
            if sections:
                section_hits[k].append(
                    1.0 if any(
                        _section_key(c.section_path, c.section_title) in sections for c in top
                    ) else 0.0
                )

        metric_rows.append(metrics)
        rows.append({
            "id": question.id,
            "question": question.question,
            "category": question.category,
            "language": question.language,
            "latency_ms": round(elapsed, 3),
            "metrics": {k: round(v, 4) for k, v in metrics.items()},
            "retrieved": [
                {
                    "rank": i + 1,
                    "chunk_id": c.chunk_id,
                    "score": c.score,
                    "source_url": c.source_url,
                    "section_path": c.section_path,
                    "relevant": bool(relevance[i]),
                }
                for i, c in enumerate(results)
            ],
        })

    summary: dict[str, Any] = {
        "evaluated_questions": len(rows),
        "skipped_no_ground_truth": skipped,
        "top_k": top_k,
        "metrics": aggregate(metric_rows),
        "section_retrieval_accuracy": {
            f"top_{k}": round(statistics.fmean(v), 4) if v else None
            for k, v in section_hits.items()
        },
        "source_retrieval_accuracy": {
            f"top_{k}": round(statistics.fmean(v), 4) if v else None
            for k, v in source_hits.items()
        },
        "latency": {
            "total_retrieval": latency_summary(total_latency),
            "embedding": latency_summary(embed_latency),
            "vector_search": latency_summary(search_latency),
        },
    }
    return RetrievalEvalResult(rows, summary)


@dataclass
class RagEvalResult:
    per_query: list[dict[str, Any]]
    summary: dict[str, Any]


def evaluate_rag(
    rag: RagService,
    dataset: EvalDataset,
    top_k: Optional[int] = None,
    judge: Optional[LLMJudge] = None,
) -> RagEvalResult:
    rows: list[dict[str, Any]] = []
    deterministic_rows: list[dict[str, float]] = []
    judge_rows: list[dict[str, float]] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "estimated_cost_usd": 0.0}
    latencies: list[float] = []

    for question in dataset.questions:
        answer = rag.query(question.question, top_k=top_k or settings.TOP_K)
        latencies.append(answer.latency_ms.get("total_ms", 0.0))
        if answer.usage:
            usage["input_tokens"] += answer.usage.input_tokens
            usage["output_tokens"] += answer.usage.output_tokens
            usage["estimated_cost_usd"] += answer.usage.estimated_cost_usd or 0.0

        det = deterministic_answer_metrics(
            answer, question.reference_answer, question.expect_insufficient
        )
        deterministic_rows.append({k: v for k, v in det.items() if v is not None})

        judged = None
        if judge is not None:
            result = judge.score(
                question.question, answer.answer, answer.retrieved, question.reference_answer
            )
            judged = {"scores": result.scores, "reason": result.reason, "error": result.error}
            judge_rows.append({k: v for k, v in result.scores.items() if v is not None})

        rows.append({
            "id": question.id,
            "question": question.question,
            "category": question.category,
            "language": question.language,
            "answer": answer.answer,
            "has_sufficient_context": answer.has_sufficient_context,
            "attempts": answer.attempts,
            "trace": answer.trace,
            "citations": [
                {"index": c.index, "source_url": c.source_url, "section_path": c.section_path}
                for c in answer.citations
            ],
            "deterministic": det,
            "judge": judged,
            "latency_ms": answer.latency_ms,
        })

    if judge is not None:
        usage["input_tokens"] += judge.usage_input
        usage["output_tokens"] += judge.usage_output
        usage["estimated_cost_usd"] += judge.cost

    unavailable = _unavailable_metrics(dataset, judge is not None)

    # When generation is the offline stub, the answer-side numbers describe the
    # stub, not the system. Say so instead of presenting them as quality.
    generator_model = _generator_model(rag)
    stub_generation = generator_model.startswith("echo")
    if stub_generation:
        for key in (
            "citation_validity", "citation_correctness_lexical", "citation_completeness",
            "lexical_groundedness", "answer_token_f1", "insufficient_context_handling",
        ):
            unavailable[key] = (
                "generation ran on the offline stub (LLM_PROVIDER=echo); "
                "set a real LLM_PROVIDER and API key to measure this"
            )

    attempt_counts = [len(r.get("attempts") or []) for r in rows]
    agentic_rows = [n for n in attempt_counts if n]
    summary = {
        "evaluated_questions": len(rows),
        "generator_model": generator_model,
        "generation_is_stub": stub_generation,
        "agentic": {
            "enabled": bool(agentic_rows),
            "mean_attempts": (
                round(statistics.fmean(agentic_rows), 3) if agentic_rows else None
            ),
            "questions_needing_a_rewrite": sum(1 for n in agentic_rows if n > 1),
            "questions_exhausting_attempts": sum(
                1 for r in rows
                if (r.get("attempts") and not r["has_sufficient_context"])
            ),
        },
        "deterministic_metrics": aggregate(deterministic_rows),
        "judge_metrics": aggregate(judge_rows) if judge_rows else {},
        "judge_enabled": judge is not None,
        "judge_model": judge.llm.model if judge else None,
        "unavailable_metrics": unavailable,
        "latency": latency_summary(latencies),
        "llm_usage": {
            **usage,
            "total_tokens": usage["input_tokens"] + usage["output_tokens"],
            "estimated_cost_usd": round(usage["estimated_cost_usd"], 6),
        },
    }
    return RagEvalResult(rows, summary)


def _generator_model(answerer) -> str:
    """Find the generation model through either orchestrator shape."""
    for path in (("generator", "llm"), ("_workflow", "generator", "llm")):
        node = answerer
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if node is not None:
            return getattr(node, "model", "")
    return ""


def _unavailable_metrics(dataset: EvalDataset, judge_enabled: bool) -> dict[str, str]:
    out: dict[str, str] = {}
    if not any(q.reference_answer for q in dataset.questions):
        out["answer_correctness"] = "no reference_answer in evaluation dataset"
        out["answer_token_f1"] = "no reference_answer in evaluation dataset"
    if not any(q.expect_insufficient is not None for q in dataset.questions):
        out["insufficient_context_handling"] = "no expect_insufficient flags in dataset"
    if not judge_enabled:
        for key in JUDGE_KEYS:
            out.setdefault(key, "LLM judge disabled (no LLM credentials or --no-judge)")
    return out


def write_report(payload: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path
