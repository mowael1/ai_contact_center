"""Context evaluation and query rewriting, driven by a separate judge LLM.

These were extension points while the pipeline was linear. The agentic graph
(:mod:`rag.graph.workflow`) now needs real implementations for its branch and
loop, so they live here.

The judge is deliberately a *different* model from the generator
(``JUDGE_LLM_MODEL``, default ``gemini-2.5-flash``, versus
``LLM_MODEL``, default ``gemini-2.5-pro``): the model that writes the answer
should not be the one deciding whether its evidence was good enough.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional, Sequence

from rag.config import settings
from rag.llm.base import LLMService
from rag.logging_utils import get_logger
from rag.models import RetrievedChunk
from rag.services.interfaces import ContextVerdict
from rag.text_utils import detect_language

logger = get_logger(__name__)

EVALUATOR_SYSTEM = """You grade retrieved knowledge-base passages for a Vodafone Egypt
support assistant. You do not answer the question yourself.

Judge only whether the passages contain enough concrete information to answer the
question correctly and specifically. Passages that are merely on-topic, or that
describe a related but different product, are NOT sufficient.

Reply with a single JSON object and nothing else."""

EVALUATOR_PROMPT = """Question:
{question}

Retrieved passages:
{context}

Return JSON with exactly these keys:
- "is_relevant": true if the passages can answer the question, false otherwise
- "confidence": 0.0-1.0, how certain you are
- "reason": one short sentence
- "missing": what information is absent (empty string if nothing is missing)
"""

REWRITER_SYSTEM = """You rewrite failed search queries for a Vodafone Egypt knowledge
base of Arabic and English web pages.

The previous query did not retrieve usable passages. Write ONE better search query.

Rules:
- Keep the user's original intent exactly. Do not answer the question.
- Add the concrete product/service vocabulary the knowledge base is likely to use
  (for example: Flex, RED, Fakka, Vodafone Cash, DSL, bundle, باقة, تجديد).
- If the question is Arabic, include key Arabic terms; adding the English product
  name alongside is good, because the corpus mixes both.
- Drop conversational filler and keep it under 25 words.

Reply with the rewritten query only - no quotes, no explanation, no preamble."""

REWRITER_PROMPT = """Original question: {question}
Previous query that failed: {previous_query}
Attempt number: {attempt}
{missing_block}
Rewritten search query:"""


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


def _format_context(chunks: Sequence[RetrievedChunk], limit: int = 5000) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, 1):
        path = " > ".join(chunk.section_path or []) or "(no section)"
        blocks.append(f"[{i}] {path}\n{chunk.text}")
    return "\n\n".join(blocks)[:limit] or "(no passages retrieved)"


class LLMContextEvaluator:
    """Judges whether retrieved context can answer the question."""

    def __init__(self, judge_llm: LLMService, threshold: Optional[float] = None) -> None:
        self.llm = judge_llm
        self.threshold = (
            threshold if threshold is not None else settings.CONTEXT_RELEVANCE_THRESHOLD
        )
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

    def evaluate(self, query: str, chunks: Sequence[RetrievedChunk]) -> ContextVerdict:
        if not chunks:
            return ContextVerdict(
                is_relevant=False, reason="no passages retrieved", confidence=1.0
            )
        try:
            response = self.llm.generate(
                system=EVALUATOR_SYSTEM,
                prompt=EVALUATOR_PROMPT.format(
                    question=query, context=_format_context(chunks)
                ),
                max_tokens=400,
                temperature=0.0,
            )
        except Exception as exc:
            # A judge outage must not block answering: fail open, and say so.
            logger.warning("Context evaluator failed (%s); treating context as relevant",
                           type(exc).__name__)
            return ContextVerdict(
                is_relevant=True, reason=f"evaluator unavailable ({type(exc).__name__})",
                confidence=0.0,
            )

        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.cost += response.usage.estimated_cost_usd or 0.0

        parsed = _parse_json(response.text)
        if parsed is None:
            logger.warning("Context evaluator returned unparseable output; failing open")
            return ContextVerdict(True, "unparseable evaluator output", 0.0)

        try:
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        is_relevant = bool(parsed.get("is_relevant", False))
        # A "relevant" verdict the judge is not confident about does not pass.
        if is_relevant and confidence < self.threshold:
            is_relevant = False

        return ContextVerdict(
            is_relevant=is_relevant,
            reason=str(parsed.get("reason", ""))[:300],
            confidence=confidence,
            missing=str(parsed.get("missing", ""))[:300],
        )


class LLMQueryRewriter:
    """Produces an improved query after a failed retrieval attempt."""

    def __init__(self, judge_llm: LLMService) -> None:
        self.llm = judge_llm
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost = 0.0

    def rewrite(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        attempt: int = 1,
        original_question: Optional[str] = None,
        missing: str = "",
    ) -> str:
        missing_block = f"Information that was missing: {missing}\n" if missing else ""
        try:
            response = self.llm.generate(
                system=REWRITER_SYSTEM,
                prompt=REWRITER_PROMPT.format(
                    question=original_question or query,
                    previous_query=query,
                    attempt=attempt,
                    missing_block=missing_block,
                ),
                max_tokens=120,
                temperature=0.3,  # a little variation, else retries repeat themselves
            )
        except Exception as exc:
            logger.warning("Query rewriter failed (%s); reusing previous query",
                           type(exc).__name__)
            return query

        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        self.cost += response.usage.estimated_cost_usd or 0.0

        rewritten = (response.text or "").strip().strip('"').strip()
        rewritten = rewritten.split("\n")[0].strip()
        if not rewritten or len(rewritten) < 3:
            return query
        if detect_language(rewritten) == "unknown":
            return query
        logger.info("Query rewritten (attempt %d): %r -> %r", attempt, query, rewritten)
        return rewritten
