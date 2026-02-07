"""Embedding cache: LRU with hash-based dedup.

Wraps any EmbeddingModel to avoid re-embedding identical texts.
Inspired by Mastra's embedding cache pattern.

Usage:
    from qortex.vec.cache import CachedEmbeddingModel
    from qortex.vec.embeddings import SentenceTransformerEmbedding

    model = CachedEmbeddingModel(SentenceTransformerEmbedding())
    # Second call with same text is instant (cache hit)
    model.embed(["hello world"])
    model.embed(["hello world"])  # cached
"""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class CachedEmbeddingModel:
    """LRU cache wrapper around any EmbeddingModel.

    Hash-based dedup: sha256(text) → embedding vector.
    On miss: delegate to wrapped model, cache result.
    On hit: return cached embedding (no model call).

    Thread-safety: not thread-safe. Use in single-threaded contexts
    or wrap with external locking.
    """

    def __init__(self, model, max_size: int = 1000) -> None:
        self._model = model
        self._max_size = max_size
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def dimensions(self) -> int:
        return self._model.dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts with caching. Only calls model for uncached texts."""
        results: list[list[float] | None] = [None] * len(texts)
        uncached_texts: list[str] = []
        uncached_indices: list[int] = []

        # Check cache for each text
        for i, text in enumerate(texts):
            key = self._hash(text)
            if key in self._cache:
                self._cache.move_to_end(key)  # LRU touch
                results[i] = self._cache[key]
                self._hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self._misses += 1

        # Embed uncached texts in a single batch
        if uncached_texts:
            new_embeddings = self._model.embed(uncached_texts)
            for idx, text, emb in zip(uncached_indices, uncached_texts, new_embeddings):
                results[idx] = emb
                key = self._hash(text)
                self._cache[key] = emb
                # Evict oldest if over capacity
                if len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)

        return results  # type: ignore[return-value]

    @property
    def cache_stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self._max_size,
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()
