"""Provider-independent generation interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

from rag.models import LLMUsage


@dataclass(slots=True)
class LLMResponse:
    text: str
    usage: LLMUsage


@dataclass(slots=True)
class StreamChunk:
    """One incremental piece of a streamed completion.

    ``delta`` carries new text. The final chunk has ``done=True`` and carries
    the accumulated ``text`` plus token ``usage``, which most providers only
    report at the end of the stream.
    """

    delta: str = ""
    text: str = ""
    done: bool = False
    usage: Optional[LLMUsage] = None


class LLMService(ABC):
    model: str = ""

    @abstractmethod
    def generate(
        self, system: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> LLMResponse:
        """Single-turn completion."""

    def generate_stream(
        self, system: str, prompt: str, max_tokens: int = 1024, temperature: float = 0.0
    ) -> Iterator[StreamChunk]:
        """Token-by-token completion.

        The default implementation falls back to a non-streaming call and emits
        the whole answer as one chunk, so every provider satisfies the
        interface even if it has no streaming endpoint.
        """
        response = self.generate(system, prompt, max_tokens, temperature)
        yield StreamChunk(
            delta=response.text, text=response.text, done=True, usage=response.usage
        )

    @property
    def supports_streaming(self) -> bool:
        return type(self).generate_stream is not LLMService.generate_stream
