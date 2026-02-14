"""Embedding model abstractions for the vector layer.

Pluggable strategy: sentence-transformers (local), OpenAI (cloud), Ollama (local LLM).

Usage:
    # Local (default, no API key needed)
    model = SentenceTransformerEmbedding()

    # Cloud
    model = OpenAIEmbedding(api_key="sk-...")

    # Self-hosted
    model = OllamaEmbedding(model_name="nomic-embed-text")
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from qortex.observe.tracing import traced

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingModel(Protocol):
    """Protocol for embedding text into vectors.

    Implement this to plug in any embedding backend.
    """

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
    """Local embedding model backed by sentence-transformers.

    Default model: all-MiniLM-L6-v2 (384 dims, fast, solid quality).
    No API key needed — runs entirely on your machine.

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
            raise ImportError("sentence-transformers required: pip install qortex[vec]") from e
        self._model = SentenceTransformer(self._model_name)
        self._dimensions = self._model.get_sentence_embedding_dimension()
        logger.info("Loaded embedding model %s (%d dims)", self._model_name, self._dimensions)

    @traced("vec.embed.sentence_transformer", external=True)
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using sentence-transformers."""
        self._load()
        assert self._model is not None
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            span.set_attribute("embed.model", self._model_name)
            span.set_attribute("embed.batch_size", len(texts))
            span.set_attribute("embed.backend", "sentence_transformers")
        except ImportError:
            pass
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    @property
    def dimensions(self) -> int:
        self._load()
        return self._dimensions


class OpenAIEmbedding:
    """Cloud embedding model via OpenAI API.

    Default model: text-embedding-3-small (1536 dims).
    Requires: OPENAI_API_KEY env var or explicit api_key.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._client = None
        self._dimensions_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def _get_client(self):
        if self._client is not None:
            return self._client
        try:
            import openai
        except ImportError as e:
            raise ImportError("openai required: pip install openai") from e

        kwargs = {}
        if self._api_key:
            kwargs["api_key"] = self._api_key
        self._client = openai.OpenAI(**kwargs)
        return self._client

    @traced("vec.embed.openai", external=True)
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via OpenAI API."""
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            span.set_attribute("embed.model", self._model)
            span.set_attribute("embed.batch_size", len(texts))
            span.set_attribute("embed.backend", "openai")
        except ImportError:
            pass
        client = self._get_client()
        response = client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in response.data]

    @property
    def dimensions(self) -> int:
        return self._dimensions_map.get(self._model, 1536)


class OllamaEmbedding:
    """Local embedding model via Ollama.

    Runs on your machine via Ollama server. No API key needed.
    Default model: nomic-embed-text (768 dims).

    Requires: Ollama running locally (https://ollama.com)
    """

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._dimensions: int | None = None
        self._dimensions_map = {
            "nomic-embed-text": 768,
            "mxbai-embed-large": 1024,
            "all-minilm": 384,
            "snowflake-arctic-embed": 1024,
        }

    @traced("vec.embed.ollama", external=True)
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via Ollama API."""
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            span.set_attribute("embed.model", self._model_name)
            span.set_attribute("embed.batch_size", len(texts))
            span.set_attribute("embed.backend", "ollama")
        except ImportError:
            pass
        import json
        import urllib.request

        results = []
        for text in texts:
            req = urllib.request.Request(
                f"{self._base_url}/api/embeddings",
                data=json.dumps({"model": self._model_name, "prompt": text}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                emb = data["embedding"]
                results.append(emb)
                if self._dimensions is None:
                    self._dimensions = len(emb)

        return results

    @property
    def dimensions(self) -> int:
        if self._dimensions is not None:
            return self._dimensions
        return self._dimensions_map.get(self._model_name, 768)
