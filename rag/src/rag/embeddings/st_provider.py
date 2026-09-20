"""Optional local sentence-transformers backend.

Kept out of the default dependency set: it pulls in torch (~2 GB). Install
with ``pip install -r requirements-local-embeddings.txt`` when you want to run
the pipeline fully offline or benchmark a local model against a hosted one.
"""

from __future__ import annotations

from typing import Sequence

from rag.embeddings.base import EmbeddingService


class SentenceTransformerEmbedding(EmbeddingService):
    def __init__(self, model: str = "intfloat/multilingual-e5-base", batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional extra
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "pip install -r requirements-local-embeddings.txt"
            ) from exc
        self.model = model
        self._batch_size = batch_size
        self._st = SentenceTransformer(model)
        self._dim = int(self._st.get_sentence_embedding_dimension())
        # e5 models are trained with these asymmetric prefixes.
        self._needs_prefix = "e5" in model.lower()

    @property
    def dimension(self) -> int:
        return self._dim

    def _encode(self, texts: Sequence[str], prefix: str) -> list[list[float]]:
        payload = [f"{prefix}{t}" for t in texts] if self._needs_prefix else list(texts)
        vectors = self._st.encode(
            payload, batch_size=self._batch_size, normalize_embeddings=True
        )
        return [v.tolist() for v in vectors]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode(texts, "passage: ")

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], "query: ")[0]
