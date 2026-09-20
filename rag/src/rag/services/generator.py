"""Grounded answer generation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

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


@dataclass(slots=True)
class AnswerChunk:
    """One event in a streamed answer.

    Text arrives as ``delta`` events. Citations are deliberately withheld until
    ``done``: they are resolved from the finished answer's bracketed indices
    against retrieved metadata, so they cannot be emitted early and cannot be
    invented mid-stream.
    """

    delta: str = ""
    text: str = ""
    done: bool = False
    answer: Optional[RagAnswer] = None
    #: Character offset this event rewrites from. 0 or absent means "append".
    #: Only the final reconciliation event ever sets it below what was emitted.
    replaces_from: Optional[int] = None


def _common_prefix_len(a: str, b: str) -> int:
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[i] == b[i]:
        i += 1
    return i


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

    def generate_stream(
        self, query: str, chunks: Sequence[RetrievedChunk]
    ) -> Iterator[AnswerChunk]:
        """Stream the answer token by token, then emit the finished RagAnswer.

        The ``INSUFFICIENT_CONTEXT`` marker is suppressed from the visible
        stream so a refusal never flashes the internal sentinel at the user.
        """
        t0 = time.perf_counter()
        context = build_context(chunks, self.max_context_chars)

        if context.is_empty:
            answer = self._no_context_answer(query, chunks)
            yield AnswerChunk(delta=answer.answer, text=answer.answer)
            yield AnswerChunk(text=answer.answer, done=True, answer=answer)
            return

        raw_parts: list[str] = []
        visible = ""
        usage = None
        # The marker can straddle two stream frames ("INSUFFICIENT" + "_CONTEXT"),
        # so hold back a tail long enough to complete it before emitting.
        hold = len(INSUFFICIENT_MARKER) - 1
        for piece in self.llm.generate_stream(
            system=SYSTEM_PROMPT,
            prompt=self._user_prompt(query, context.text),
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        ):
            if piece.usage is not None:
                usage = piece.usage
            if piece.delta:
                raw_parts.append(piece.delta)
                raw_so_far = "".join(raw_parts)
                cleaned = raw_so_far.replace(INSUFFICIENT_MARKER, "")
                emittable = cleaned[:-hold] if hold and len(cleaned) > hold else ""
                delta = emittable[len(visible):]
                if delta:
                    visible = emittable
                    yield AnswerChunk(delta=delta, text=visible)
            if piece.done and piece.text:
                raw_parts = [piece.text]

        answer = self._finalize(
            query, "".join(raw_parts), chunks, context.citations, usage,
            (time.perf_counter() - t0) * 1000,
        )
        # Flush whatever was held back. Post-processing (marker removal, invalid
        # citation stripping) can rewrite the tail, so reconcile on the common
        # prefix rather than assuming the stream is a prefix of the final text.
        common = _common_prefix_len(visible, answer.answer)
        if common < len(answer.answer):
            yield AnswerChunk(
                delta=answer.answer[common:],
                text=answer.answer,
                replaces_from=common,
            )
        yield AnswerChunk(text=answer.answer, done=True, answer=answer)

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
