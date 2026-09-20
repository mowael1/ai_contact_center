"""``google/embeddinggemma-300m`` embeddings.

EmbeddingGemma is a 300M-parameter multilingual retrieval model (100+
languages, including Arabic) producing 768-dimensional vectors.

Two backends are provided because the model can be run either way:

* :class:`HFInferenceEmbeddingGemma` - HuggingFace's hosted inference endpoint.
  No torch, no weights on disk. Needs an HF token and the model's licence
  accepted on the model page (it is a gated repo).
* :class:`LocalEmbeddingGemma` - local sentence-transformers. Fully offline,
  but needs torch plus ~1.2 GB of weights.

Both apply EmbeddingGemma's **mandatory task prefixes**. The model is trained
asymmetrically and retrieval quality drops sharply without them:

    query    ->  "task: search result | query: {text}"
    document ->  "title: none | text: {text}"
"""

from __future__ import annotations

import os
import time
from typing import Any, Sequence

import httpx

from rag.config import settings
from rag.embeddings.base import EmbeddingService
from rag.logging_utils import get_logger

logger = get_logger(__name__)

MODEL_ID = "google/embeddinggemma-300m"
DIMENSION = 768

QUERY_PREFIX = "task: search result | query: "
DOCUMENT_PREFIX = "title: none | text: "

RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def format_query(text: str) -> str:
    return f"{QUERY_PREFIX}{text}"


def format_document(text: str, title: str | None = None) -> str:
    return f"title: {title or 'none'} | text: {text}"


class HFInferenceEmbeddingGemma(EmbeddingService):
    """EmbeddingGemma over the HuggingFace Inference API."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        batch_size: int | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.model = model or MODEL_ID
        self._api_key = (
            api_key or settings.HF_API_TOKEN or os.getenv("HF_TOKEN", "")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN", "")
        )
        if not self._api_key:
            raise RuntimeError(
                "HFInferenceEmbeddingGemma requires an HF token. Set HF_API_TOKEN in "
                "rag/.env (see rag/.env.example). google/embeddinggemma-300m is a "
                "gated repo, so accept its licence on the model page first."
            )
        # The hosted endpoint is far happier with modest batches than the
        # 96-item default used for the big commercial embedding APIs.
        self._batch_size = batch_size or min(settings.EMBEDDING_BATCH_SIZE, 32)
        self._dim = DIMENSION
        self._client = httpx.Client(timeout=timeout)
        self._url = (
            f"{settings.HF_INFERENCE_ENDPOINT.rstrip('/')}"
            f"/models/{self.model}/pipeline/feature-extraction"
        )

    @property
    def dimension(self) -> int:
        return self._dim

    def _post(self, inputs: list[str]) -> Any:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {"inputs": inputs, "options": {"wait_for_model": True}}
        delay = 2.0
        last: Exception | None = None

        for attempt in range(4):
            try:
                response = self._client.post(self._url, headers=headers, json=payload)
                if response.status_code in RETRY_STATUS:
                    raise httpx.HTTPStatusError(
                        f"transient {response.status_code}",
                        request=response.request, response=response,
                    )
                if response.status_code in (401, 403):
                    raise RuntimeError(
                        "HuggingFace rejected the token for "
                        f"{self.model}. It is a gated model: accept its licence on "
                        "the model page and use a token with read access."
                    )
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last = exc
                logger.warning(
                    "EmbeddingGemma request failed (%s), attempt %d/4",
                    type(exc).__name__, attempt + 1,
                )
                if attempt < 3:
                    time.sleep(delay)
                    delay *= 2
        raise RuntimeError("EmbeddingGemma request failed after 4 attempts") from last

    @staticmethod
    def _as_vectors(payload: Any, expected: int) -> list[list[float]]:
        """Normalise the feature-extraction response to one vector per input.

        The endpoint may return [batch][dim] for a pooled model, or
        [batch][tokens][dim] if pooling is not applied - mean-pool in that case.
        """
        if not isinstance(payload, list) or not payload:
            raise RuntimeError("Unexpected empty response from the embedding endpoint")

        first = payload[0]
        if isinstance(first, (int, float)):  # single input, already pooled
            return [list(map(float, payload))]
        if isinstance(first, list) and first and isinstance(first[0], (int, float)):
            return [list(map(float, row)) for row in payload]
        if isinstance(first, list) and first and isinstance(first[0], list):
            pooled: list[list[float]] = []
            for tokens in payload:
                width = len(tokens[0])
                sums = [0.0] * width
                for token in tokens:
                    for i, value in enumerate(token):
                        sums[i] += value
                pooled.append([s / len(tokens) for s in sums])
            return pooled
        raise RuntimeError("Could not interpret the embedding endpoint response shape")

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            vectors = self._as_vectors(self._post(batch), len(batch))
            out.extend(vectors)
        if out:
            self._dim = len(out[0])
        return out

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed([format_document(t) for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._embed([format_query(text)])[0]

    def model_info(self) -> dict[str, Any]:
        return {
            "provider": "HFInferenceEmbeddingGemma",
            "model": self.model,
            "dimension": self.dimension,
            "backend": "huggingface_inference_api",
            "task_prefixes": {"query": QUERY_PREFIX, "document": DOCUMENT_PREFIX},
        }


class LocalEmbeddingGemma(EmbeddingService):
    """EmbeddingGemma run locally through sentence-transformers.

    Requires ``pip install -r requirements-local-embeddings.txt`` (torch) and
    roughly 1.2 GB of free disk for the weights.
    """

    def __init__(self, model: str | None = None, batch_size: int = 16) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "sentence-transformers is not installed. Run "
                "pip install -r rag/requirements-local-embeddings.txt, or use "
                "EMBEDDING_PROVIDER=embeddinggemma_hf to call the hosted API instead."
            ) from exc
        self.model = model or MODEL_ID
        self._batch_size = batch_size
        # sentence-transformers ships EmbeddingGemma prompt templates, so the
        # task prefixes are applied via prompt_name rather than by hand.
        self._st = SentenceTransformer(self.model)
        self._dim = int(self._st.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._st.encode_document(
            list(texts), batch_size=self._batch_size, normalize_embeddings=True
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self._st.encode_query([text], normalize_embeddings=True)[0].tolist()

    def model_info(self) -> dict[str, Any]:
        return {
            "provider": "LocalEmbeddingGemma",
            "model": self.model,
            "dimension": self.dimension,
            "backend": "sentence_transformers_local",
        }
