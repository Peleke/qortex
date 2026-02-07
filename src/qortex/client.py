"""QortexClient: the public interface for qortex memory.

All consumer adapters (crewai, langchain, agno, Mastra, MCP) target this
protocol. Third parties build their own adapters against it.

Architecture:
    QortexClient (Protocol)
    ├── LocalQortexClient  — direct in-process, no MCP subprocess
    └── McpQortexClient    — talks to qortex MCP server (future)

Result types have conversion helpers for framework interop:
    QueryItem.to_langchain_document()
    QueryItem.to_crewai_result()
    QueryItem.to_agno_document()
    QueryItem.to_mastra_result()
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class QueryItem:
    """A single result from a qortex query.

    Canonical shape: {id, content, score, domain, node_id, metadata}
    All framework adapters map FROM this shape.
    """

    id: str
    content: str
    score: float
    domain: str
    node_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Framework conversion helpers --

    def to_langchain_document(self) -> Any:
        """Convert to langchain Document(page_content, metadata, id).

        Returns a dict matching langchain's Document shape. If langchain
        is installed, returns an actual Document instance.
        """
        meta = {"score": self.score, "domain": self.domain, "node_id": self.node_id}
        meta.update(self.metadata)
        try:
            from langchain_core.documents import Document

            return Document(page_content=self.content, metadata=meta, id=self.id)
        except ImportError:
            return {"page_content": self.content, "metadata": meta, "id": self.id}

    def to_crewai_result(self) -> dict[str, Any]:
        """Convert to crewai SearchResult shape: {id, content, metadata, score}."""
        meta = {"domain": self.domain, "node_id": self.node_id}
        meta.update(self.metadata)
        return {
            "id": self.id,
            "content": self.content,
            "metadata": meta,
            "score": self.score,
        }

    def to_agno_document(self) -> Any:
        """Convert to agno Document shape.

        Returns a dict matching agno's Document. If agno is installed,
        returns an actual Document instance.
        """
        meta = {"domain": self.domain, "node_id": self.node_id}
        meta.update(self.metadata)
        try:
            from agno.document import Document

            return Document(
                content=self.content,
                id=self.id,
                name=self.node_id,
                meta_data=meta,
                reranking_score=self.score,
            )
        except ImportError:
            return {
                "content": self.content,
                "id": self.id,
                "name": self.node_id,
                "meta_data": meta,
                "reranking_score": self.score,
            }

    def to_mastra_result(self) -> dict[str, Any]:
        """Convert to Mastra QueryResult shape: {id, score, metadata, document}."""
        meta = {"domain": self.domain, "node_id": self.node_id}
        meta.update(self.metadata)
        return {
            "id": self.id,
            "score": self.score,
            "metadata": meta,
            "document": self.content,
        }


@dataclass
class QueryResult:
    """Result of a qortex query."""

    items: list[QueryItem]
    query_id: str


@dataclass
class FeedbackResult:
    """Result of submitting feedback."""

    status: str
    query_id: str
    outcome_count: int
    source: str


@dataclass
class IngestResult:
    """Result of ingesting a file."""

    domain: str
    source: str
    concepts: int
    edges: int
    rules: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class DomainInfo:
    """Information about a domain."""

    name: str
    description: str | None = None
    concept_count: int = 0
    edge_count: int = 0
    rule_count: int = 0


@dataclass
class StatusResult:
    """Server/client status."""

    status: str
    backend: str
    vector_index: str | None = None
    vector_search: bool = False
    graph_algorithms: bool = False
    domain_count: int = 0
    embedding_model: str | None = None


# ---------------------------------------------------------------------------
# QortexClient protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class QortexClient(Protocol):
    """The public interface for qortex memory.

    Python consumers implement this directly or use LocalQortexClient.
    Non-Python consumers use the MCP server (same semantics).
    Third parties build framework adapters against this protocol.
    """

    def query(
        self,
        context: str,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
    ) -> QueryResult: ...

    def feedback(
        self,
        query_id: str,
        outcomes: dict[str, str],
        source: str = "unknown",
    ) -> FeedbackResult: ...

    def ingest(
        self,
        source_path: str,
        domain: str,
        source_type: str | None = None,
    ) -> IngestResult: ...

    def domains(self) -> list[DomainInfo]: ...

    def status(self) -> StatusResult: ...


# ---------------------------------------------------------------------------
# LocalQortexClient — direct in-process, no MCP
# ---------------------------------------------------------------------------


class LocalQortexClient:
    """In-process QortexClient backed by VectorIndex + GraphBackend.

    No MCP subprocess, no network. Composes vec and graph layers directly.
    This is the reference implementation — all MCP tools delegate to the
    same logic.

    When mode="graph" (or "auto" with graph available), uses GraphRAGAdapter
    for graph-enhanced retrieval with teleportation factors + online edge gen.
    When mode="vec", uses VecOnlyAdapter (pure cosine similarity).
    """

    def __init__(
        self,
        vector_index: Any,
        backend: Any,
        embedding_model: Any = None,
        llm_backend: Any = None,
        mode: str = "auto",
    ) -> None:
        from qortex.hippocampus.adapter import GraphRAGAdapter, VecOnlyAdapter

        self._vector_index = vector_index
        self._backend = backend
        self._embedding_model = embedding_model
        self._llm_backend = llm_backend
        self._mode = mode

        if vector_index is not None and embedding_model is not None:
            if mode == "graph" or (mode == "auto" and self._has_graph_edges()):
                self._adapter = GraphRAGAdapter(vector_index, backend, embedding_model)
            else:
                self._adapter = VecOnlyAdapter(vector_index, backend, embedding_model)
        else:
            self._adapter = None

    def _has_graph_edges(self) -> bool:
        """Check if the backend has any edges (worth running PPR)."""
        try:
            from qortex.core.memory import InMemoryBackend
            if isinstance(self._backend, InMemoryBackend):
                return len(self._backend._edges) > 0
        except ImportError:
            pass
        return self._backend.supports_mage()

    def query(
        self,
        context: str,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
    ) -> QueryResult:
        # Clamp to safe ranges (top_k: [1, 1000], min_confidence: [0.0, 1.0])
        top_k = max(1, min(top_k, 1000))
        min_confidence = max(0.0, min(min_confidence, 1.0))

        if self._adapter is None:
            return QueryResult(items=[], query_id=str(uuid.uuid4()))

        result = self._adapter.retrieve(
            query=context, domains=domains, top_k=top_k, min_confidence=min_confidence
        )

        items = [
            QueryItem(
                id=item.id,
                content=item.content,
                score=round(item.score, 4),
                domain=item.domain,
                node_id=item.node_id or item.id,
                metadata=item.metadata,
            )
            for item in result.items
        ]
        return QueryResult(items=items, query_id=result.query_id)

    def feedback(
        self,
        query_id: str,
        outcomes: dict[str, str],
        source: str = "unknown",
    ) -> FeedbackResult:
        if self._adapter is not None:
            self._adapter.feedback(query_id, outcomes)

        return FeedbackResult(
            status="recorded",
            query_id=query_id,
            outcome_count=len(outcomes),
            source=source,
        )

    def ingest(
        self,
        source_path: str,
        domain: str,
        source_type: str | None = None,
    ) -> IngestResult:
        from pathlib import Path

        from qortex_ingest.base import Source

        path = Path(source_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {source_path}")

        allowed = {"text", "markdown", "pdf"}
        if source_type is not None and source_type not in allowed:
            raise ValueError(f"Invalid source_type: {source_type}. Must be one of {allowed}")

        if source_type is None:
            ext = path.suffix.lower()
            type_map = {".md": "markdown", ".markdown": "markdown", ".pdf": "pdf", ".txt": "text"}
            source_type = type_map.get(ext, "text")

        source = Source(path=path, source_type=source_type, name=path.name)
        llm = self._get_llm_backend()

        if source_type == "markdown":
            from qortex_ingest.markdown import MarkdownIngestor

            ingestor = MarkdownIngestor(llm, embedding_model=self._embedding_model)
        elif source_type == "pdf":
            from qortex_ingest.pdf import PDFIngestor

            ingestor = PDFIngestor(llm, embedding_model=self._embedding_model)
        else:
            from qortex_ingest.text import TextIngestor

            ingestor = TextIngestor(llm, embedding_model=self._embedding_model)

        manifest = ingestor.ingest(source, domain=domain)
        self._backend.ingest_manifest(manifest)

        # Dual-write embeddings
        ids_with_embeddings = []
        embeddings_list = []
        for concept in manifest.concepts:
            if concept.embedding is not None:
                self._backend.add_embedding(concept.id, concept.embedding)
                ids_with_embeddings.append(concept.id)
                embeddings_list.append(concept.embedding)

        if self._vector_index is not None and ids_with_embeddings:
            self._vector_index.add(ids_with_embeddings, embeddings_list)

        return IngestResult(
            domain=domain,
            source=path.name,
            concepts=len(manifest.concepts),
            edges=len(manifest.edges),
            rules=len(manifest.rules),
            warnings=manifest.warnings,
        )

    def domains(self) -> list[DomainInfo]:
        return [
            DomainInfo(
                name=d.name,
                description=d.description,
                concept_count=d.concept_count,
                edge_count=d.edge_count,
                rule_count=d.rule_count,
            )
            for d in self._backend.list_domains()
        ]

    def status(self) -> StatusResult:
        return StatusResult(
            status="ok",
            backend=type(self._backend).__name__,
            vector_index=type(self._vector_index).__name__ if self._vector_index else None,
            vector_search=self._vector_index is not None,
            graph_algorithms=self._backend.supports_mage() if self._backend else False,
            domain_count=len(self._backend.list_domains()) if self._backend else 0,
            embedding_model=(
                getattr(self._embedding_model, "_model_name", type(self._embedding_model).__name__)
                if self._embedding_model
                else None
            ),
        )

    def _get_llm_backend(self):
        if self._llm_backend is not None:
            return self._llm_backend

        try:
            from qortex_ingest.backends.anthropic import AnthropicBackend

            self._llm_backend = AnthropicBackend()
            return self._llm_backend
        except ImportError:
            logger.debug("anthropic backend not installed, falling back to StubLLMBackend")
        except Exception:
            logger.warning(
                "Failed to initialize AnthropicBackend, falling back to StubLLMBackend",
                exc_info=True,
            )

        from qortex_ingest.base import StubLLMBackend

        self._llm_backend = StubLLMBackend()
        return self._llm_backend
