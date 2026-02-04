"""Graph backend protocol and implementations.

The KG is backend-agnostic. Code against GraphBackend protocol.
Primary: Memgraph (for MAGE algorithms)
Fallback: SQLite (limited features, no PPR)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Iterator, Literal, Protocol, runtime_checkable

from .models import ConceptEdge, ConceptNode, Domain, IngestionManifest

if TYPE_CHECKING:
    from .memgraph import MemgraphBackend as MemgraphBackendImpl


class GraphPattern:
    """Structured query pattern (alternative to raw Cypher)."""
    # TODO: Define query DSL
    pass


@runtime_checkable
class GraphBackend(Protocol):
    """Backend-agnostic graph interface.

    Implementations: MemgraphBackend, SQLiteBackend
    """

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to backend."""
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if backend is available."""
        ...

    # -------------------------------------------------------------------------
    # Domain Management (Schema-like isolation)
    # -------------------------------------------------------------------------

    @abstractmethod
    def create_domain(self, name: str, description: str | None = None) -> Domain:
        """Create a new domain (isolated subgraph)."""
        ...

    @abstractmethod
    def get_domain(self, name: str) -> Domain | None:
        """Get domain by name."""
        ...

    @abstractmethod
    def list_domains(self) -> list[Domain]:
        """List all domains."""
        ...

    @abstractmethod
    def delete_domain(self, name: str) -> bool:
        """Delete domain and all its contents."""
        ...

    # -------------------------------------------------------------------------
    # Node Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def add_node(self, node: ConceptNode) -> None:
        """Add a concept node to its domain."""
        ...

    @abstractmethod
    def get_node(self, node_id: str, domain: str | None = None) -> ConceptNode | None:
        """Get node by ID. If domain is None, search all domains."""
        ...

    @abstractmethod
    def find_nodes(
        self,
        domain: str | None = None,
        name_pattern: str | None = None,
        limit: int = 100,
    ) -> Iterator[ConceptNode]:
        """Find nodes matching criteria."""
        ...

    # -------------------------------------------------------------------------
    # Edge Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def add_edge(self, edge: ConceptEdge) -> None:
        """Add an edge between concepts."""
        ...

    @abstractmethod
    def get_edges(
        self,
        node_id: str,
        direction: Literal["in", "out", "both"] = "both",
        relation_type: str | None = None,
    ) -> Iterator[ConceptEdge]:
        """Get edges connected to a node."""
        ...

    # -------------------------------------------------------------------------
    # Manifest Ingestion (Atomic bulk load)
    # -------------------------------------------------------------------------

    @abstractmethod
    def ingest_manifest(self, manifest: IngestionManifest) -> None:
        """Atomically ingest a full manifest into the KG.

        This is the main entry point from ingestors.
        Should be transactional where possible.
        """
        ...

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    @abstractmethod
    def query(self, pattern: GraphPattern) -> Iterator[dict]:
        """Execute a structured query."""
        ...

    @abstractmethod
    def query_cypher(self, cypher: str, params: dict | None = None) -> Iterator[dict]:
        """Execute raw Cypher query (Memgraph/Neo4j compatible).

        Use sparingly - prefer structured queries.
        """
        ...

    # -------------------------------------------------------------------------
    # Graph Algorithms (MAGE or fallback)
    # -------------------------------------------------------------------------

    @abstractmethod
    def supports_mage(self) -> bool:
        """Whether this backend supports MAGE algorithms."""
        ...

    @abstractmethod
    def personalized_pagerank(
        self,
        source_nodes: list[str],
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        domain: str | None = None,
    ) -> dict[str, float]:
        """Personalized PageRank from source nodes.

        Core algorithm for HippoRAG pattern completion.
        Returns node_id -> score mapping.
        """
        ...

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    @abstractmethod
    def checkpoint(self, name: str, domains: list[str] | None = None) -> str:
        """Create a named checkpoint. Returns checkpoint ID."""
        ...

    @abstractmethod
    def restore(self, checkpoint_id: str) -> None:
        """Restore to a checkpoint state."""
        ...

    @abstractmethod
    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints."""
        ...


# =============================================================================
# Backend Factory
# =============================================================================


def get_backend(
    backend_type: Literal["memgraph", "sqlite"] = "memgraph",
    **kwargs,
) -> GraphBackend:
    """Get a graph backend instance.

    Args:
        backend_type: "memgraph" (primary) or "sqlite" (fallback)
        **kwargs: Backend-specific configuration

    Returns:
        Connected GraphBackend instance
    """
    if backend_type == "memgraph":
        from .memgraph import MemgraphBackend
        return MemgraphBackend(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 7687),
            username=kwargs.get("username", ""),
            password=kwargs.get("password", ""),
        )
    elif backend_type == "sqlite":
        raise NotImplementedError("SQLite backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def get_connected_backend(
    backend_type: Literal["memgraph", "sqlite"] = "memgraph",
    **kwargs,
) -> GraphBackend:
    """Get a connected graph backend instance.

    Convenience function that also calls connect().
    """
    backend = get_backend(backend_type, **kwargs)
    backend.connect()
    return backend
