"""Graph backend protocol and implementations.

The KG is backend-agnostic. Code against GraphBackend protocol.
Primary: Memgraph (for MAGE algorithms)
Fallback: SQLite (limited features, no PPR)
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Iterator, Literal, Protocol, runtime_checkable

from .models import ConceptEdge, ConceptNode, Domain, IngestionManifest


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
# Stub Implementations (to be filled in)
# =============================================================================


class MemgraphBackend:
    """Primary backend using Memgraph + MAGE.

    TODO: Implement with gqlalchemy or neo4j driver
    """

    def __init__(self, host: str = "localhost", port: int = 7687):
        self.host = host
        self.port = port
        self._connection = None

    def connect(self) -> None:
        raise NotImplementedError("M1: Implement Memgraph connection")

    def disconnect(self) -> None:
        raise NotImplementedError()

    def is_connected(self) -> bool:
        return False

    def create_domain(self, name: str, description: str | None = None) -> Domain:
        raise NotImplementedError()

    def get_domain(self, name: str) -> Domain | None:
        raise NotImplementedError()

    def list_domains(self) -> list[Domain]:
        raise NotImplementedError()

    def delete_domain(self, name: str) -> bool:
        raise NotImplementedError()

    def add_node(self, node: ConceptNode) -> None:
        raise NotImplementedError()

    def get_node(self, node_id: str, domain: str | None = None) -> ConceptNode | None:
        raise NotImplementedError()

    def find_nodes(
        self,
        domain: str | None = None,
        name_pattern: str | None = None,
        limit: int = 100,
    ) -> Iterator[ConceptNode]:
        raise NotImplementedError()

    def add_edge(self, edge: ConceptEdge) -> None:
        raise NotImplementedError()

    def get_edges(
        self,
        node_id: str,
        direction: Literal["in", "out", "both"] = "both",
        relation_type: str | None = None,
    ) -> Iterator[ConceptEdge]:
        raise NotImplementedError()

    def ingest_manifest(self, manifest: IngestionManifest) -> None:
        raise NotImplementedError()

    def query(self, pattern: GraphPattern) -> Iterator[dict]:
        raise NotImplementedError()

    def query_cypher(self, cypher: str, params: dict | None = None) -> Iterator[dict]:
        raise NotImplementedError()

    def supports_mage(self) -> bool:
        return True

    def personalized_pagerank(
        self,
        source_nodes: list[str],
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        domain: str | None = None,
    ) -> dict[str, float]:
        raise NotImplementedError("M3: Implement PPR via MAGE")

    def checkpoint(self, name: str, domains: list[str] | None = None) -> str:
        raise NotImplementedError()

    def restore(self, checkpoint_id: str) -> None:
        raise NotImplementedError()

    def list_checkpoints(self) -> list[dict]:
        raise NotImplementedError()


class SQLiteBackend:
    """Fallback backend using SQLite adjacency list.

    Limited features - no MAGE algorithms.
    Use for environments without Memgraph.
    """

    def __init__(self, db_path: str = "~/.qortex/graph.db"):
        self.db_path = db_path
        self._connection = None

    def supports_mage(self) -> bool:
        return False

    def personalized_pagerank(
        self,
        source_nodes: list[str],
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        domain: str | None = None,
    ) -> dict[str, float]:
        """Simple BFS-based approximation (not true PPR)."""
        raise NotImplementedError("M3: Implement simple traversal fallback")

    # ... other methods similar stubs ...


def get_backend(prefer_memgraph: bool = True) -> GraphBackend:
    """Get the best available backend.

    Tries Memgraph first, falls back to SQLite.
    """
    if prefer_memgraph:
        backend = MemgraphBackend()
        try:
            backend.connect()
            return backend
        except Exception:
            pass

    # Fallback
    return SQLiteBackend()
