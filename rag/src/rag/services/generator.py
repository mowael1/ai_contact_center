"""Grounded answer generation."""

from __future__ import annotations

import time
from typing import Optional, Sequence

from rag.config import settings
from rag.llm.base import LLMService
from rag.logging_utils import get_logger
from rag.models import Citation, RagAnswer, RetrievedChunk
from rag.services.citations import resolve_citations, strip_invalid_markers
from rag.services.context_builder import build_context
from rag.services.prompts import (
    INSUFFICIENT_MARKER,
    LANGUAGE_INSTRUCTIONS,
    NO_CONTEXT_ANSWERS,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from rag.text_utils import detect_language

logger = get_logger(__name__)


class GenerationService:
    def __init__(self, llm: LLMService, max_context_chars: Optional[int] = None):
        self.llm = llm
        self.max_context_chars = max_context_chars or settings.MAX_CONTEXT_CHARS

    @staticmethod
    def _user_prompt(query: str, context_text: str) -> str:
        """Resolve the answer language deterministically from the question.

        Smaller models tend to answer in the language of the *context* rather
        than the question - measured on gemini-2.5-flash-lite, an English
        question over Arabic passages came back in Arabic. Naming the target
        language explicitly fixes it without relying on the model to infer it.
        """
        language = detect_language(query)
        return USER_PROMPT.format(
            context=context_text,
            question=query,
            language_instruction=LANGUAGE_INSTRUCTIONS.get(
                language, LANGUAGE_INSTRUCTIONS["unknown"]
            ),
        )

    def _finalize(
        self,
        query: str,
        raw: str,
        chunks: Sequence[RetrievedChunk],
        context_citations: list[Citation],
        usage,
        elapsed_ms: float,
    ) -> RagAnswer:
        """Shared post-processing for both the blocking and streaming paths."""
        sufficient = INSUFFICIENT_MARKER not in raw
        answer = raw.replace(INSUFFICIENT_MARKER, "").strip()
        valid = {c.index for c in context_citations}
        answer = strip_invalid_markers(answer, valid)
        citations = (
            resolve_citations(raw, context_citations, keep_all=True) if sufficient else []
        )
        logger.info(
            "Generated answer (%d chars, sufficient=%s, %d citations)",
            len(answer), sufficient, len(citations),
        )
        return RagAnswer(
            query=query,
            answer=answer,
            citations=citations,
            retrieved=list(chunks),
            has_sufficient_context=sufficient,
            usage=usage,
            latency_ms={"generate_ms": round(elapsed_ms, 3)},
        )

    def _no_context_answer(self, query: str, chunks: Sequence[RetrievedChunk]) -> RagAnswer:
        language = detect_language(query)
        return RagAnswer(
            query=query,
            answer=NO_CONTEXT_ANSWERS.get(language, NO_CONTEXT_ANSWERS["en"]),
            citations=[],
            retrieved=list(chunks),
            has_sufficient_context=False,
            latency_ms={"generate_ms": 0.0},
        )

    def generate(self, query: str, chunks: Sequence[RetrievedChunk]) -> RagAnswer:
        t0 = time.perf_counter()
        context = build_context(chunks, self.max_context_chars)

        if context.is_empty:
            return self._no_context_answer(query, chunks)

        response = self.llm.generate(
            system=SYSTEM_PROMPT,
            prompt=self._user_prompt(query, context.text),
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )
        return self._finalize(
            query, response.text or "", chunks, context.citations, response.usage,
            (time.perf_counter() - t0) * 1000,
        )
