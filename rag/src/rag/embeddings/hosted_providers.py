"""Hosted multilingual embedding providers (OpenAI / Cohere).

Both are accessed over plain ``httpx`` rather than a vendor SDK: the surface we
need is one endpoint, and it keeps the dependency list and the abstraction
small, as required by the design brief.
"""

from __future__ import annotations

import os
import time
from typing import Sequence

import httpx

from rag.config import settings
from rag.embeddings.base import EmbeddingService
from rag.logging_utils import get_logger

logger = get_logger(__name__)

# Known output dimensions, used before the first call is made.
_KNOWN_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "embed-multilingual-v3.0": 1024,
    "embed-multilingual-light-v3.0": 384,
}

RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class _HttpEmbeddingBase(EmbeddingService):
    def __init__(self, model: str, api_key: str, batch_size: int, timeout: float = 60.0):
        if not api_key:
            raise RuntimeError(
                f"{type(self).__name__} requires an API key. Set it in rag/.env "
                "(see .env.example). No credential is ever read from code."
            )
        self.model = model
        self._api_key = api_key
        self._batch_size = batch_size
        self._dim = _KNOWN_DIMENSIONS.get(model, 0)
        self._client = httpx.Client(timeout=timeout)

    @property
    def dimension(self) -> int:
        return self._dim

    def _post_with_retry(self, url: str, headers: dict, payload: dict, attempts: int = 4):
        delay = 1.0
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._client.post(url, headers=headers, json=payload)
                if response.status_code in RETRY_STATUS:
                    raise httpx.HTTPStatusError(
                        f"transient status {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last = exc
                # The message is logged without the payload so no key leaks.
                logger.warning(
                    "Embedding request failed (attempt %d/%d): %s",
                    attempt + 1, attempts, type(exc).__name__,
                )
                if attempt < attempts - 1:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError(f"Embedding request failed after {attempts} attempts") from last

    def _embed_batched(self, texts: Sequence[str], is_query: bool) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            out.extend(self._embed_batch(batch, is_query))
        if out and not self._dim:
            self._dim = len(out[0])
        return out

    def _embed_batch(self, batch: list[str], is_query: bool) -> list[list[float]]:
        raise NotImplementedError

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed_batched(texts, is_query=False)

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batched([text], is_query=True)[0]


class OpenAIEmbedding(_HttpEmbeddingBase):
    """``text-embedding-3-*`` - strong multilingual performance incl. Arabic."""

    def __init__(self, model: str | None = None, api_key: str | None = None, **kw):
        super().__init__(
            model or settings.EMBEDDING_MODEL,
            api_key or settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", ""),
            kw.pop("batch_size", settings.EMBEDDING_BATCH_SIZE),
        )
        self._base = (settings.OPENAI_BASE_URL or "https://api.openai.com/v1").rstrip("/")

    def _embed_batch(self, batch: list[str], is_query: bool) -> list[list[float]]:
        data = self._post_with_retry(
            f"{self._base}/embeddings",
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            {"model": self.model, "input": batch},
        )
        rows = sorted(data["data"], key=lambda r: r["index"])
        return [r["embedding"] for r in rows]


class CohereEmbedding(_HttpEmbeddingBase):
    """``embed-multilingual-v3.0`` - explicitly trained for Arabic + English.

    Cohere distinguishes ``search_document`` from ``search_query`` inputs,
    which is why ``is_query`` is threaded through the base class.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None, **kw):
        super().__init__(
            model or settings.EMBEDDING_MODEL,
            api_key or settings.COHERE_API_KEY or os.getenv("COHERE_API_KEY", ""),
            kw.pop("batch_size", min(settings.EMBEDDING_BATCH_SIZE, 96)),
        )

    def _embed_batch(self, batch: list[str], is_query: bool) -> list[list[float]]:
        data = self._post_with_retry(
            "https://api.cohere.com/v2/embed",
            {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            {
                "model": self.model,
                "texts": batch,
                "input_type": "search_query" if is_query else "search_document",
                "embedding_types": ["float"],
            },
        )
        embeddings = data.get("embeddings", {})
        return embeddings.get("float", embeddings) if isinstance(embeddings, dict) else embeddings
