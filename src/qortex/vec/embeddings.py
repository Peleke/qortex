"""Embedding model abstractions for the vector layer."""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding text into vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (same length as texts).
        """
        ...

    @property
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...


class SentenceTransformerEmbedding:
    """Embedding model backed by sentence-transformers.

    Default model: all-MiniLM-L6-v2 (384 dims, fast, solid quality).
    Requires: pip install qortex[vec]
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model = None
        self._dimensions: int | None = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required: pip install qortex[vec]"
            ) from e
        self._model = SentenceTransformer(self._model_name)
        self._dimensions = self._model.get_sentence_embedding_dimension()
        logger.info("Loaded embedding model %s (%d dims)", self._model_name, self._dimensions)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using sentence-transformers."""
        self._load()
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimensions(self) -> int:
        """Dimensionality of the embedding vectors."""
        self._load()
        return self._dimensions
