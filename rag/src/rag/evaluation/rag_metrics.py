"""RAG answer-quality metrics.

Two clearly separated families:

* **Deterministic** - citation validity/completeness, lexical grounding,
  insufficient-context handling, token-overlap answer correctness. Reproducible
  and free.
* **LLM-as-a-judge** - faithfulness, answer relevance, context relevance,
  context precision/recall, answer correctness, unsupported-claim rate. These
  call a model and are reported under a separate ``judge`` key so nobody
  mistakes them for deterministic numbers.

Any metric whose ground truth is absent from the evaluation dataset is
reported as ``None`` ("unavailable"), never as a fabricated score.
"""

from __future__ import annotations

import json
import re
import statistics
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from rag.llm.base import LLMService
from rag.logging_utils import get_logger
from rag.models import Citation, RagAnswer, RetrievedChunk
from rag.services.citations import extract_cited_indices
from rag.text_utils import normalize_text

logger = get_logger(__name__)

_TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)
# Sentences carrying a number, code or price are the claims worth citing.
_FACTUAL = re.compile(r"\d|[*#]\d|%|جنيه|EGP|GB|MB", re.IGNORECASE)


def tokens(text: str) -> set[str]:
    return set(_TOKEN.findall(normalize_text(text or "").lower()))


# --------------------------------------------------------------------------
# Deterministic
# --------------------------------------------------------------------------
def citation_validity(answer: RagAnswer) -> Optional[float]:
    """Share of the model's bracketed references that map to a real passage.

    Returns ``None`` when the answer cited nothing (the metric is undefined),
    so it is not silently counted as a perfect score.
    """
    cited = extract_cited_indices(answer.answer)
    if not cited:
        return None
    valid = {i + 1 for i in range(len(answer.retrieved))}
    return sum(1 for c in cited if c in valid) / len(cited)


def citation_correctness(answer: RagAnswer, overlap_threshold: float = 0.12) -> Optional[float]:
    """Do the cited passages lexically support the answer's content?

    A deterministic proxy: the share of cited passages whose token overlap with
    the answer clears ``overlap_threshold``. Weaker than an LLM judge but
    reproducible and free.
    """
    if not answer.citations:
        return None
    answer_tokens = tokens(answer.answer)
    if not answer_tokens:
        return 0.0
    by_id = {c.chunk_id: c for c in answer.retrieved}
    scores = []
    for citation in answer.citations:
        chunk = by_id.get(citation.chunk_id)
        if chunk is None:
            scores.append(0.0)
            continue
        chunk_tokens = tokens(chunk.text)
        if not chunk_tokens:
            scores.append(0.0)
            continue
        scores.append(
            1.0 if len(answer_tokens & chunk_tokens) / len(answer_tokens) >= overlap_threshold
            else 0.0
        )
    return round(statistics.fmean(scores), 4) if scores else None


def citation_completeness(answer: RagAnswer) -> Optional[float]:
    """Share of factual sentences (containing a number/price/code) that carry a citation."""
    sentences = [s for s in re.split(r"(?<=[.!?؟])\s+", answer.answer or "") if s.strip()]
    factual = [s for s in sentences if _FACTUAL.search(s)]
    if not factual:
        return None
    cited = sum(1 for s in factual if re.search(r"\[\d{1,2}\]", s))
    return round(cited / len(factual), 4)


def lexical_groundedness(answer: RagAnswer) -> Optional[float]:
    """Share of the answer's content tokens that also occur in the context.

    A cheap, deterministic floor on faithfulness: a high unsupported-token rate
    is strong evidence of hallucination, though a high score does not by itself
    prove faithfulness (hence the LLM judge alongside it).
    """
    answer_tokens = tokens(answer.answer)
    if not answer_tokens:
        return None
    context_tokens: set[str] = set()
    for chunk in answer.retrieved:
        context_tokens |= tokens(chunk.text)
    if not context_tokens:
        return 0.0
    return round(len(answer_tokens & context_tokens) / len(answer_tokens), 4)


def answer_token_f1(answer: str, reference: Optional[str]) -> Optional[float]:
    """Token-level F1 against a reference answer. ``None`` if no reference."""
    if not reference:
        return None
    a, r = tokens(answer), tokens(reference)
    if not a or not r:
        return 0.0
    overlap = len(a & r)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(r)
    return round(2 * precision * recall / (precision + recall), 4)


def insufficient_context_handling(
    answer: RagAnswer, expected_insufficient: Optional[bool]
) -> Optional[float]:
    """1.0 when the system's refusal behaviour matches the dataset's expectation."""
    if expected_insufficient is None:
        return None
    return 1.0 if answer.has_sufficient_context != expected_insufficient else 0.0


def deterministic_answer_metrics(
    answer: RagAnswer,
    reference_answer: Optional[str] = None,
    expected_insufficient: Optional[bool] = None,
) -> dict[str, Optional[float]]:
    return {
        "citation_validity": citation_validity(answer),
        "citation_correctness_lexical": citation_correctness(answer),
        "citation_completeness": citation_completeness(answer),
        "lexical_groundedness": lexical_groundedness(answer),
        "answer_token_f1": answer_token_f1(answer.answer, reference_answer),
        "insufficient_context_handling": insufficient_context_handling(
            answer, expected_insufficient
        ),
    }


# --------------------------------------------------------------------------
# LLM as a judge
# --------------------------------------------------------------------------
JUDGE_SYSTEM = """You are a strict evaluator of retrieval-augmented answers.
Score only what you are shown. Respond with a single JSON object and nothing else.
Scores are floats from 0.0 to 1.0."""

JUDGE_PROMPT = """Question:
{question}

Retrieved context passages:
{context}

Generated answer:
{answer}

{reference_block}
Return JSON with exactly these keys:
- "faithfulness": is every claim in the answer supported by the context?
- "answer_relevance": does the answer address the question?
- "context_relevance": how relevant is the retrieved context to the question?
- "context_precision": what share of the retrieved context is actually useful?
- "context_recall": does the context contain what is needed to answer fully?
- "answer_correctness": correctness vs the reference answer, or null if no reference given.
- "unsupported_claim_rate": share of factual claims NOT supported by context (0.0 = all supported).
- "reason": one short sentence.
"""

JUDGE_KEYS = (
    "faithfulness", "answer_relevance", "context_relevance", "context_precision",
    "context_recall", "answer_correctness", "unsupported_claim_rate",
)


@dataclass(slots=True)
class JudgeResult:
    scores: dict[str, Optional[float]]
    reason: str = ""
    error: Optional[str] = None


class LLMJudge:
    """LLM-as-a-judge scorer. Results are always reported separately from
    deterministic metrics."""

    def __init__(self, llm: LLMService, max_context_chars: int = 6000):
        self.llm = llm
        self.max_context_chars = max_context_chars
        self.usage_input = 0
        self.usage_output = 0
        self.cost = 0.0

    def score(
        self,
        question: str,
        answer: str,
        chunks: Sequence[RetrievedChunk],
        reference_answer: Optional[str] = None,
    ) -> JudgeResult:
        context = "\n\n".join(
            f"[{i + 1}] {c.text}" for i, c in enumerate(chunks)
        )[: self.max_context_chars]
        reference_block = (
            f"Reference answer:\n{reference_answer}\n" if reference_answer else
            "No reference answer is available; set answer_correctness to null.\n"
        )
        try:
            response = self.llm.generate(
                system=JUDGE_SYSTEM,
                prompt=JUDGE_PROMPT.format(
                    question=question, context=context or "(none)",
                    answer=answer or "(empty)", reference_block=reference_block,
                ),
                max_tokens=600,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Judge call failed: %s", type(exc).__name__)
            return JudgeResult({k: None for k in JUDGE_KEYS}, error=type(exc).__name__)

        self.usage_input += response.usage.input_tokens
        self.usage_output += response.usage.output_tokens
        self.cost += response.usage.estimated_cost_usd or 0.0

        parsed = _parse_json(response.text)
        if parsed is None:
            return JudgeResult({k: None for k in JUDGE_KEYS}, error="unparseable_judge_output")
        scores = {k: _as_float(parsed.get(k)) for k in JUDGE_KEYS}
        return JudgeResult(scores, reason=str(parsed.get("reason", ""))[:300])


def _parse_json(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
