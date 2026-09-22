"""Dependency-free embedding providers for tests and offline development.

Neither is suitable for production retrieval. They exist so that the chunking
algorithm, the CLI and the test-suite can run without network access or API
credentials.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Sequence

from rag.embeddings.base import EmbeddingService

_TOKEN = re.compile(r"[\w؀-ۿ]+", re.UNICODE)


def _l2(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector] if norm else vector


class HashEmbedding(EmbeddingService):
    """Deterministic pseudo-random vectors. Similarity is meaningless.

    Used only to assert plumbing in unit tests.
    """

    def __init__(self, model: str = "hash-256", dim: int = 256) -> None:
        self.model = model
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vector = [
            (digest[i % len(digest)] ^ (i * 31 % 256)) / 255.0 - 0.5
            for i in range(self._dim)
        ]
        return _l2(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)


class LocalLexicalEmbedding(EmbeddingService):
    """Hashed character-n-gram + word bag-of-features vectors.

    Unlike :class:`HashEmbedding` the cosine similarity it produces is a real
    (lexical, not semantic) signal, which makes it usable for validating the
    semantic chunker's boundary behaviour offline. Character n-grams give it
    partial robustness to Arabic morphology.
    """

    def __init__(self, model: str = "local-lexical-512", dim: int = 512, ngram: int = 4) -> None:
        self.model = model
        self._dim = dim
        self._ngram = ngram

    @property
    def dimension(self) -> int:
        return self._dim

    def _features(self, text: str) -> Counter:
        text = text.lower()
        features: Counter = Counter()
        for word in _TOKEN.findall(text):
            features[f"w:{word}"] += 2.0
            padded = f" {word} "
            for i in range(len(padded) - self._ngram + 1):
                features[f"c:{padded[i : i + self._ngram]}"] += 1.0
        return features

    def _one(self, text: str) -> list[float]:
        vector = [0.0] * self._dim
        for feature, weight in self._features(text).items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            # Sub-linear term weighting keeps one repeated token from dominating.
            vector[index] += sign * (1.0 + math.log(weight))
        return _l2(vector)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._one(text)
