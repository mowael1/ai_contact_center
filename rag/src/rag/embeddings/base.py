"""Embedding provider interface.

The retriever, the chunker and the evaluator all talk to this interface only,
so swapping model or vendor is a config change (see README "Embedding
experiments").
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class EmbeddingService(ABC):
    """Abstract multilingual embedding backend."""

    #: Human-readable model identifier, echoed into diagnostics.
    model: str = ""

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of passages."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality."""

    def model_info(self) -> dict[str, Any]:
        return {
            "provider": type(self).__name__,
            "model": self.model,
            "dimension": self.dimension,
        }
