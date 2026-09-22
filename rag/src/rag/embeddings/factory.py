"""Provider selection and an on-disk-free in-process embedding cache."""

from __future__ import annotations

from typing import Optional, Sequence

from rag.config import settings
from rag.embeddings.base import EmbeddingService
from rag.logging_utils import get_logger
from rag.text_utils import content_hash

logger = get_logger(__name__)


class CachedEmbedding(EmbeddingService):
    """Memoises embeddings by content hash so unchanged text is never re-sent.

    This matters twice over: the semantic chunker embeds every sentence, and
    re-ingestion of an unchanged document would otherwise repay the full API
    cost.
    """

    def __init__(self, inner: EmbeddingService, max_entries: int = 200_000) -> None:
        self._inner = inner
        self._cache: dict[str, list[float]] = {}
        self._max = max_entries
        self.model = inner.model
        self.hits = 0
        self.misses = 0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [content_hash(self.model, t) for t in texts]
        missing_idx = [i for i, k in enumerate(keys) if k not in self._cache]
        self.hits += len(texts) - len(missing_idx)
        self.misses += len(missing_idx)
        if missing_idx:
            fresh = self._inner.embed_documents([texts[i] for i in missing_idx])
            for i, vector in zip(missing_idx, fresh):
                if len(self._cache) < self._max:
                    self._cache[keys[i]] = vector
                else:
                    self._cache[keys[i]] = vector  # keep correctness over bound
        return [self._cache[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        return self._inner.embed_query(text)

    def model_info(self) -> dict:
        info = self._inner.model_info()
        info.update({"cached": True, "cache_hits": self.hits, "cache_misses": self.misses})
        return info


def build_embedding_service(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    cache: Optional[bool] = None,
) -> EmbeddingService:
    """Instantiate the configured embedding backend."""
    provider = (provider or settings.EMBEDDING_PROVIDER).lower()
    model = model or settings.EMBEDDING_MODEL

    service: EmbeddingService
    if provider in ("embeddinggemma_hf", "embeddinggemma", "gemma_hf"):
        from rag.embeddings.gemma_provider import HFInferenceEmbeddingGemma

        service = HFInferenceEmbeddingGemma(model=model)
    elif provider in ("hf_inference", "hf"):
        from rag.embeddings.gemma_provider import HFInferenceEmbedding

        service = HFInferenceEmbedding(model=model)
    elif provider in ("embeddinggemma_local", "gemma_local"):
        from rag.embeddings.gemma_provider import LocalEmbeddingGemma

        service = LocalEmbeddingGemma(model=model)
    elif provider == "openai":
        from rag.embeddings.hosted_providers import OpenAIEmbedding

        service = OpenAIEmbedding(model=model)
    elif provider == "cohere":
        from rag.embeddings.hosted_providers import CohereEmbedding

        service = CohereEmbedding(model=model)
    elif provider == "sentence_transformers":
        from rag.embeddings.st_provider import SentenceTransformerEmbedding

        service = SentenceTransformerEmbedding(model=model)
    elif provider in ("local_lexical", "lexical"):
        from rag.embeddings.local_providers import LocalLexicalEmbedding

        service = LocalLexicalEmbedding()
    elif provider == "hash":
        from rag.embeddings.local_providers import HashEmbedding

        service = HashEmbedding()
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")

    logger.info("Embedding provider=%s model=%s", provider, service.model)
    use_cache = settings.EMBEDDING_CACHE_ENABLED if cache is None else cache
    return CachedEmbedding(service) if use_cache else service
