"""LangChain VectorStore: qortex as a drop-in VectorStore.

This is the full VectorStore integration, not just a retriever.
Use this anywhere LangChain expects a VectorStore: chains, agents,
as_retriever(), similarity_search(), add_documents(), etc.

Usage:
    from qortex.adapters.langchain_vectorstore import QortexVectorStore

    # From an existing client
    vs = QortexVectorStore(client=client, domain="security")

    # From texts (creates everything for you)
    vs = QortexVectorStore.from_texts(
        texts=["OAuth2 is...", "JWT tokens..."],
        embedding=my_embedding,
        metadatas=[{"source": "docs"}, {"source": "docs"}],
        domain="security",
    )

    # Standard VectorStore API
    docs = vs.similarity_search("authentication", k=5)
    docs_and_scores = vs.similarity_search_with_score("authentication", k=5)
    retriever = vs.as_retriever(search_kwargs={"k": 10})

    # qortex extras: graph exploration + rules
    explore = vs.explore(docs[0].metadata["node_id"])
    rules = vs.rules(concept_ids=[d.metadata["node_id"] for d in docs])
    vs.feedback({"item_id": "accepted"})

Requires: pip install langchain-core
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from qortex.client import ExploreResult, RulesResult


def _check_langchain() -> None:
    if not _HAS_LANGCHAIN:
        raise ImportError(
            "langchain-core required for QortexVectorStore: pip install langchain-core"
        )


if _HAS_LANGCHAIN:

    class QortexVectorStore(VectorStore):
        """qortex as a LangChain VectorStore.

        Wraps QortexClient and exposes the full VectorStore interface:
        similarity_search, add_texts, from_texts, as_retriever, etc.

        Unlike plain vector stores, qortex results carry graph structure:
        - Documents include node_id in metadata for graph navigation
        - explore() traverses typed edges from any retrieved concept
        - rules() returns projected rules linked to retrieved concepts
        - feedback() closes the learning loop (teleportation factor updates)
        """

        def __init__(
            self,
            client: Any,
            domain: str = "default",
            feedback_source: str = "langchain",
        ) -> None:
            """Initialize QortexVectorStore.

            Args:
                client: A QortexClient (LocalQortexClient or future McpQortexClient).
                domain: Default domain for searches and add operations.
                feedback_source: Source identifier for feedback events.
            """
            self._client = client
            self._domain = domain
            self._feedback_source = feedback_source
            self._last_query_id: str | None = None

        @property
        def embeddings(self) -> Embeddings | None:
            """Access the embedding model if available."""
            emb = getattr(self._client, "embedding_model", None)
            if emb is None:
                return None
            # Wrap qortex embedding model in LangChain Embeddings interface
            if isinstance(emb, Embeddings):
                return emb
            return _QortexEmbeddingsWrapper(emb)

        def add_texts(
            self,
            texts: Iterable[str],
            metadatas: list[dict] | None = None,
            *,
            ids: list[str] | None = None,
            **kwargs: Any,
        ) -> list[str]:
            """Add texts to the vector store.

            Creates ConceptNodes in the backend and indexes their embeddings.
            Each text becomes a concept in the configured domain.

            Args:
                texts: Texts to add.
                metadatas: Optional metadata dicts per text.
                ids: Optional IDs. Auto-generated if not provided.
                **kwargs: domain (str) to override the default domain.

            Returns:
                List of IDs of added concepts.
            """
            domain = kwargs.get("domain", self._domain)
            return self._client.add_concepts(
                texts=list(texts),
                domain=domain,
                metadatas=metadatas,
                ids=ids,
            )

        def similarity_search(
            self, query: str, k: int = 4, **kwargs: Any
        ) -> list[Document]:
            """Return docs most similar to query.

            Args:
                query: Input text.
                k: Number of documents to return.
                **kwargs: Additional args (domains, min_confidence).

            Returns:
                List of langchain Document objects.
            """
            docs_and_scores = self.similarity_search_with_score(query, k, **kwargs)
            return [doc for doc, _ in docs_and_scores]

        def similarity_search_with_score(
            self, query: str, k: int = 4, **kwargs: Any
        ) -> list[tuple[Document, float]]:
            """Return docs and similarity scores.

            Args:
                query: Input text.
                k: Number of documents to return.
                **kwargs: Additional args.

            Returns:
                List of (Document, score) tuples. Score is 0-1 similarity.
            """
            domains = kwargs.get("domains", [self._domain])
            min_confidence = kwargs.get("min_confidence", 0.0)

            result = self._client.query(
                context=query, domains=domains, top_k=k, min_confidence=min_confidence,
            )
            self._last_query_id = result.query_id

            docs_and_scores = []
            for item in result.items:
                meta = {
                    "score": item.score,
                    "domain": item.domain,
                    "node_id": item.node_id,
                }
                meta.update(item.metadata)

                # Include rules in metadata if present
                if result.rules:
                    meta["rules"] = [
                        {"id": r.id, "text": r.text, "relevance": r.relevance}
                        for r in result.rules
                        if item.node_id in r.source_concepts
                    ]

                doc = Document(page_content=item.content, metadata=meta, id=item.id)
                docs_and_scores.append((doc, item.score))

            return docs_and_scores

        def _select_relevance_score_fn(self):
            """qortex scores are already cosine similarity in [0, 1]."""
            return lambda score: score

        @classmethod
        def from_texts(
            cls,
            texts: list[str],
            embedding: Embeddings,
            metadatas: list[dict] | None = None,
            *,
            ids: list[str] | None = None,
            **kwargs: Any,
        ) -> QortexVectorStore:
            """Create a QortexVectorStore from texts.

            Sets up a full qortex backend (InMemoryBackend + NumpyVectorIndex),
            adds the texts, and returns a ready-to-use VectorStore.

            Args:
                texts: Texts to add.
                embedding: LangChain Embeddings instance.
                metadatas: Optional metadata per text.
                ids: Optional IDs.
                **kwargs: domain (str), backend, vector_index for advanced use.

            Returns:
                Initialized QortexVectorStore.
            """
            domain = kwargs.pop("domain", "default")
            backend = kwargs.pop("backend", None)
            vector_index = kwargs.pop("vector_index", None)

            # Wrap LangChain embedding in qortex interface
            qortex_embedding = _LangChainEmbeddingWrapper(embedding)

            if backend is None:
                from qortex.core.memory import InMemoryBackend
                from qortex.vec.index import NumpyVectorIndex

                if vector_index is None:
                    vector_index = NumpyVectorIndex(dimensions=qortex_embedding.dimensions)
                backend = InMemoryBackend(vector_index=vector_index)
                backend.connect()

            from qortex.client import LocalQortexClient

            client = LocalQortexClient(
                vector_index=vector_index,
                backend=backend,
                embedding_model=qortex_embedding,
            )

            store = cls(client=client, domain=domain)
            store.add_texts(texts, metadatas, ids=ids)
            return store

        def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
            """Get documents by their node IDs."""
            nodes = self._client.get_nodes(list(ids))
            docs = []
            for node in nodes:
                meta = {
                    "domain": node.domain,
                    "node_id": node.id,
                }
                meta.update(node.properties)
                docs.append(Document(
                    page_content=f"{node.name}: {node.description}",
                    metadata=meta,
                    id=node.id,
                ))
            return docs

        # -- qortex extras: graph exploration + rules --

        def explore(self, node_id: str, depth: int = 1) -> ExploreResult | None:
            """Explore a concept's graph neighborhood.

            Call this on a node_id from search results to see typed edges,
            neighbors, and linked rules.
            """
            return self._client.explore(node_id, depth)

        def rules(self, **kwargs: Any) -> RulesResult:
            """Get projected rules from the knowledge graph.

            Args:
                **kwargs: Passed to client.rules() — domains, concept_ids,
                    categories, include_derived, min_confidence.
            """
            return self._client.rules(**kwargs)

        def feedback(self, outcomes: dict[str, str]) -> None:
            """Report feedback for the last search. Closes the learning loop."""
            if self._last_query_id is None:
                return
            self._client.feedback(
                query_id=self._last_query_id,
                outcomes=outcomes,
                source=self._feedback_source,
            )

        @property
        def last_query_id(self) -> str | None:
            return self._last_query_id


    class _QortexEmbeddingsWrapper(Embeddings):
        """Wrap a qortex embedding model in LangChain's Embeddings interface."""

        def __init__(self, qortex_model: Any) -> None:
            self._model = qortex_model

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._model.embed(texts)

        def embed_query(self, text: str) -> list[float]:
            return self._model.embed([text])[0]


    class _LangChainEmbeddingWrapper:
        """Wrap a LangChain Embeddings instance in qortex's embed() interface."""

        def __init__(self, lc_embedding: Embeddings) -> None:
            self._lc = lc_embedding
            # Probe dimensions
            probe = lc_embedding.embed_query("test")
            self._dimensions = len(probe)

        @property
        def dimensions(self) -> int:
            return self._dimensions

        def embed(self, texts: list[str]) -> list[list[float]]:
            return self._lc.embed_documents(texts)


else:

    class QortexVectorStore:  # type: ignore[no-redef]
        """Stub: langchain-core not installed."""

        def __init__(self, **kwargs: Any) -> None:
            _check_langchain()
