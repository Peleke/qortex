"""Vector layer for qortex: embeddings + similarity search.

Install extras:
    pip install qortex[vec]          # numpy + sentence-transformers
    pip install qortex[vec-sqlite]   # + sqlite-vec for persistent index

Usage:
    from qortex.vec import SentenceTransformerEmbedding, NumpyVectorIndex

    model = SentenceTransformerEmbedding()
    index = NumpyVectorIndex(dimensions=model.dimensions)

    vecs = model.embed(["hello world", "goodbye"])
    index.add(["id1", "id2"], vecs)
    results = index.search(model.embed(["hi"])[0], top_k=1)
"""

from qortex.vec.cache import CachedEmbeddingModel
from qortex.vec.embeddings import (
    EmbeddingModel,
    OllamaEmbedding,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
)
from qortex.vec.index import NumpyVectorIndex, VectorIndex

__all__ = [
    "CachedEmbeddingModel",
    "EmbeddingModel",
    "NumpyVectorIndex",
    "OllamaEmbedding",
    "OpenAIEmbedding",
    "SentenceTransformerEmbedding",
    "VectorIndex",
]
