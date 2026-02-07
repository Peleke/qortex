"""MCP server implementation for qortex.

Tools exposed:
- qortex_query: Retrieve relevant knowledge items for a context
- qortex_feedback: Report outcomes for retrieved items
- qortex_ingest: Ingest a file into a domain
- qortex_domains: List available domains
- qortex_status: Server health + backend info
- qortex_explore: Explore a node's neighborhood in the graph
- qortex_rules: Get projected rules from the knowledge graph

Architecture:
    Each tool has a plain `_<name>_impl()` function with the core logic,
    plus an `@mcp.tool`-decorated wrapper that delegates to it.
    Tests call the `_impl` functions directly; MCP clients hit the wrappers.

    Vec layer and graph layer are independent:
        QORTEX_VEC=memory|sqlite   → VectorIndex (similarity search)
        QORTEX_GRAPH=memory|memgraph → GraphBackend (node metadata, PPR, rules)
    The adapter composes both layers.
"""

from __future__ import annotations

import atexit
import logging
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from qortex.core.memory import InMemoryBackend
from qortex.hippocampus.adapter import VecOnlyAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server-level state: initialized once via create_server() or serve()
# ---------------------------------------------------------------------------

_backend: Any = None  # GraphBackend: node metadata, domains, rules, PPR
_vector_index: Any = None  # VectorIndex: similarity search (independent)
_adapter: Any = None  # VecOnlyAdapter (Level 0)
_graph_adapter: Any = None  # GraphRAGAdapter (Level 1+)
_embedding_model: Any = None
_interoception: Any = None  # InteroceptionProvider: feedback lifecycle

mcp = FastMCP("qortex")


def _shutdown_interoception() -> None:
    """atexit handler: persist interoception state."""
    if _interoception is not None:
        try:
            _interoception.shutdown()
        except Exception:
            logger.warning("Failed to shut down interoception layer", exc_info=True)


def _ensure_initialized() -> None:
    """Lazy initialization: set up vec + graph layers from env config."""
    global _backend, _vector_index, _adapter, _embedding_model, _interoception

    if _backend is not None:
        return

    # --- Embedding model ---
    try:
        from qortex.vec.embeddings import SentenceTransformerEmbedding

        _embedding_model = SentenceTransformerEmbedding()
    except ImportError:
        logger.warning("qortex[vec] not installed. Vector search unavailable.")
        _embedding_model = None

    # --- Vec layer (independent of graph) ---
    if _embedding_model is not None:
        vec_backend = os.environ.get("QORTEX_VEC", "sqlite")
        if vec_backend == "sqlite":
            try:
                from qortex.vec.index import SqliteVecIndex

                vec_path = Path("~/.qortex/vectors.db").expanduser()
                _vector_index = SqliteVecIndex(
                    db_path=str(vec_path), dimensions=_embedding_model.dimensions
                )
            except ImportError:
                logger.warning(
                    "sqlite-vec not installed, using in-memory vector index. "
                    "Install qortex[vec-sqlite] for persistence."
                )
                from qortex.vec.index import NumpyVectorIndex

                _vector_index = NumpyVectorIndex(dimensions=_embedding_model.dimensions)
        else:  # "memory"
            from qortex.vec.index import NumpyVectorIndex

            _vector_index = NumpyVectorIndex(dimensions=_embedding_model.dimensions)

    # --- Graph layer (independent of vec) ---
    graph_backend = os.environ.get("QORTEX_GRAPH", "memory")
    if graph_backend == "memgraph":
        try:
            from qortex.core.backend import MemgraphBackend

            host = os.environ.get("MEMGRAPH_HOST", "localhost")
            port = int(os.environ.get("MEMGRAPH_PORT", "7687"))
            _backend = MemgraphBackend(host=host, port=port)
            _backend.connect()
        except (ImportError, Exception) as e:
            logger.warning("Memgraph unavailable (%s), falling back to InMemoryBackend.", e)
            _backend = InMemoryBackend(vector_index=_vector_index)
            _backend.connect()
    else:  # "memory"
        _backend = InMemoryBackend(vector_index=_vector_index)
        _backend.connect()

    # --- Interoception layer (feedback lifecycle) ---
    from qortex.hippocampus.interoception import InteroceptionConfig, LocalInteroceptionProvider

    qortex_dir = Path("~/.qortex").expanduser()
    interoception_config = InteroceptionConfig(
        factors_path=qortex_dir / "factors.json",
        buffer_path=qortex_dir / "edge_buffer.json",
    )
    _interoception = LocalInteroceptionProvider(interoception_config)
    _interoception.startup()
    atexit.register(_shutdown_interoception)

    # --- Compose adapters ---
    if _vector_index is not None and _embedding_model is not None:
        _adapter = VecOnlyAdapter(_vector_index, _backend, _embedding_model)

        # Graph adapter: GraphRAGAdapter for Level 1+ retrieval
        from qortex.hippocampus.adapter import GraphRAGAdapter

        _graph_adapter = GraphRAGAdapter(
            _vector_index, _backend, _embedding_model, interoception=_interoception,
        )
    else:
        _adapter = None
        _graph_adapter = None


def _select_adapter(mode: str = "auto"):
    """Select the right adapter based on mode.

    "vec":   VecOnlyAdapter (Level 0, pure cosine similarity)
    "graph": GraphRAGAdapter (Level 1+, PPR over merged graph)
    "auto":  graph if available, else vec
    """
    if mode == "vec":
        return _adapter
    if mode == "graph":
        return _graph_adapter or _adapter
    # auto: prefer graph if it exists
    return _graph_adapter or _adapter


def create_server(
    backend: Any = None,
    embedding_model: Any = None,
    vector_index: Any = None,
    interoception: Any = None,
) -> FastMCP:
    """Create and configure the MCP server with explicit layers.

    Args:
        backend: GraphBackend for node metadata, domains, rules.
        embedding_model: EmbeddingModel for query/document embedding.
        vector_index: VectorIndex for similarity search. If None and
            embedding_model is provided, creates a NumpyVectorIndex.
        interoception: InteroceptionProvider for feedback lifecycle.
            If None, creates a fresh LocalInteroceptionProvider (no persistence).

    Used by tests and advanced configurations. For normal usage, call serve().
    """
    global _backend, _vector_index, _adapter, _graph_adapter, _embedding_model, _interoception

    _backend = backend
    _embedding_model = embedding_model

    # Auto-create in-memory vector index if not provided
    if vector_index is not None:
        _vector_index = vector_index
    elif _embedding_model is not None:
        from qortex.vec.index import NumpyVectorIndex

        _vector_index = NumpyVectorIndex(dimensions=_embedding_model.dimensions)
    else:
        _vector_index = None

    # Interoception: use provided or create fresh (no persistence for tests)
    if interoception is not None:
        _interoception = interoception
    else:
        from qortex.hippocampus.interoception import LocalInteroceptionProvider
        _interoception = LocalInteroceptionProvider()

    if _vector_index is not None and _backend is not None and _embedding_model is not None:
        _adapter = VecOnlyAdapter(_vector_index, _backend, _embedding_model)

        from qortex.hippocampus.adapter import GraphRAGAdapter
        _graph_adapter = GraphRAGAdapter(
            _vector_index, _backend, _embedding_model, interoception=_interoception,
        )
    else:
        _adapter = None
        _graph_adapter = None

    return mcp


# ---------------------------------------------------------------------------
# Tool implementations (plain functions — testable, no decorator wrapping)
# ---------------------------------------------------------------------------


def _query_impl(
    context: str,
    domains: list[str] | None = None,
    top_k: int = 20,
    min_confidence: float = 0.0,
    mode: str = "auto",
) -> dict:
    _ensure_initialized()

    # Clamp inputs to valid ranges
    top_k = max(1, min(top_k, 1000))
    min_confidence = max(0.0, min(min_confidence, 1.0))

    adapter = _select_adapter(mode)
    if adapter is None:
        return {
            "items": [],
            "query_id": "",
            "error": "No embedding model available. Install qortex[vec].",
        }

    result = adapter.retrieve(
        query=context,
        domains=domains,
        top_k=top_k,
        min_confidence=min_confidence,
    )

    items = [
        {
            "id": item.id,
            "content": item.content,
            "score": round(item.score, 4),
            "domain": item.domain,
            "node_id": item.node_id,
            "metadata": item.metadata,
        }
        for item in result.items
    ]

    # Collect rules linked to activated concepts
    rules = _collect_query_rules(items, domains)

    return {
        "items": items,
        "query_id": result.query_id,
        "rules": rules,
    }


def _feedback_impl(
    query_id: str,
    outcomes: dict[str, str],
    source: str = "unknown",
) -> dict:
    _ensure_initialized()

    # Route feedback to both adapters — the graph adapter updates factors
    if _graph_adapter is not None:
        _graph_adapter.feedback(query_id, outcomes)
    elif _adapter is not None:
        _adapter.feedback(query_id, outcomes)

    return {
        "status": "recorded",
        "query_id": query_id,
        "outcome_count": len(outcomes),
        "source": source,
    }


_ALLOWED_SOURCE_TYPES = {"text", "markdown", "pdf"}


def _ingest_impl(
    source_path: str,
    domain: str,
    source_type: str | None = None,
) -> dict:
    _ensure_initialized()

    path = Path(source_path).expanduser().resolve()
    if not path.exists():
        return {"error": f"File not found: {source_path}"}

    if not path.is_file():
        return {"error": f"Not a file: {source_path}"}

    # Validate source_type if provided
    if source_type is not None and source_type not in _ALLOWED_SOURCE_TYPES:
        return {"error": f"Invalid source_type: {source_type}. Must be one of {_ALLOWED_SOURCE_TYPES}"}

    # Auto-detect source type from extension
    if source_type is None:
        ext = path.suffix.lower()
        type_map = {
            ".md": "markdown",
            ".markdown": "markdown",
            ".pdf": "pdf",
            ".txt": "text",
        }
        source_type = type_map.get(ext, "text")

    from qortex_ingest.base import Source

    source = Source(
        path=path,
        source_type=source_type,
        name=path.name,
    )

    llm = _get_llm_backend()

    if source_type == "markdown":
        from qortex_ingest.markdown import MarkdownIngestor

        ingestor = MarkdownIngestor(llm, embedding_model=_embedding_model)
    elif source_type == "pdf":
        from qortex_ingest.pdf import PDFIngestor

        ingestor = PDFIngestor(llm, embedding_model=_embedding_model)
    else:
        from qortex_ingest.text import TextIngestor

        ingestor = TextIngestor(llm, embedding_model=_embedding_model)

    manifest = ingestor.ingest(source, domain=domain)
    _backend.ingest_manifest(manifest)

    # Dual-write embeddings: VectorIndex (for search) + backend (for graph storage)
    ids_with_embeddings = []
    embeddings_list = []
    for concept in manifest.concepts:
        if concept.embedding is not None:
            _backend.add_embedding(concept.id, concept.embedding)
            ids_with_embeddings.append(concept.id)
            embeddings_list.append(concept.embedding)

    if _vector_index is not None and ids_with_embeddings:
        _vector_index.add(ids_with_embeddings, embeddings_list)

    return {
        "domain": domain,
        "source": path.name,
        "concepts": len(manifest.concepts),
        "edges": len(manifest.edges),
        "rules": len(manifest.rules),
        "warnings": manifest.warnings,
    }


def _domains_impl() -> dict:
    _ensure_initialized()

    domains = _backend.list_domains()
    return {
        "domains": [
            {
                "name": d.name,
                "description": d.description,
                "concept_count": d.concept_count,
                "edge_count": d.edge_count,
                "rule_count": d.rule_count,
            }
            for d in domains
        ]
    }


def _status_impl() -> dict:
    _ensure_initialized()

    backend_type = type(_backend).__name__
    has_vec = _vector_index is not None
    has_mage = _backend.supports_mage() if _backend else False
    domain_count = len(_backend.list_domains()) if _backend else 0

    return {
        "status": "ok",
        "backend": backend_type,
        "vector_index": type(_vector_index).__name__ if _vector_index else None,
        "vector_search": has_vec,
        "graph_algorithms": has_mage,
        "domain_count": domain_count,
        "embedding_model": (
            getattr(_embedding_model, "_model_name", type(_embedding_model).__name__)
            if _embedding_model
            else None
        ),
        "interoception": _interoception.summary() if _interoception else None,
    }


def _collect_query_rules(
    items: list[dict],
    domains: list[str] | None,
) -> list[dict]:
    """Collect rules linked to query result concepts (for MCP _query_impl)."""
    if not items or _backend is None:
        return []

    from qortex.core.rules import collect_rules_for_concepts

    activated_ids = [item["node_id"] for item in items if item.get("node_id")]
    scores_map = {item["node_id"]: item["score"] for item in items if item.get("node_id")}

    rules = collect_rules_for_concepts(_backend, activated_ids, domains, scores_map)
    return [
        {
            "id": r.id,
            "text": r.text,
            "domain": r.domain,
            "category": r.category,
            "confidence": r.confidence,
            "relevance": r.relevance,
            "derivation": r.derivation,
            "source_concepts": r.source_concepts,
            "metadata": r.metadata,
        }
        for r in rules
    ]


def _explore_impl(
    node_id: str,
    depth: int = 1,
) -> dict | None:
    """Explore a node's neighborhood in the knowledge graph."""
    _ensure_initialized()

    from collections import deque

    depth = max(1, min(depth, 3))

    node = _backend.get_node(node_id)
    if node is None:
        return None

    # BFS to collect edges and neighbors
    visited: set[str] = {node_id}
    all_edges: list[dict] = []
    all_neighbors: list[dict] = []
    frontier: deque[str] = deque([node_id])

    for _hop in range(depth):
        next_frontier: deque[str] = deque()
        while frontier:
            current_id = frontier.popleft()
            edges = list(_backend.get_edges(current_id, "both"))
            for edge in edges:
                edge_dict = {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation_type": edge.relation_type.value if hasattr(edge.relation_type, "value") else str(edge.relation_type),
                    "confidence": edge.confidence,
                    "properties": edge.properties,
                }
                all_edges.append(edge_dict)
                neighbor_id = (
                    edge.target_id if edge.source_id == current_id else edge.source_id
                )
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor_node = _backend.get_node(neighbor_id)
                    if neighbor_node is not None:
                        all_neighbors.append({
                            "id": neighbor_node.id,
                            "name": neighbor_node.name,
                            "description": neighbor_node.description,
                            "domain": neighbor_node.domain,
                            "confidence": neighbor_node.confidence,
                            "properties": neighbor_node.properties,
                        })
                        next_frontier.append(neighbor_id)
        frontier = next_frontier

    # Deduplicate edges
    seen_edges: set[tuple[str, str, str]] = set()
    unique_edges: list[dict] = []
    for e in all_edges:
        key = (e["source_id"], e["target_id"], e["relation_type"])
        if key not in seen_edges:
            seen_edges.add(key)
            unique_edges.append(e)

    # Collect rules linked to explored concepts
    from qortex.core.rules import collect_rules_for_concepts

    rules = collect_rules_for_concepts(_backend, list(visited))
    rule_dicts = [
        {
            "id": r.id,
            "text": r.text,
            "domain": r.domain,
            "category": r.category,
            "confidence": r.confidence,
            "relevance": r.relevance,
            "derivation": r.derivation,
            "source_concepts": r.source_concepts,
            "metadata": r.metadata,
        }
        for r in rules
    ]

    return {
        "node": {
            "id": node.id,
            "name": node.name,
            "description": node.description,
            "domain": node.domain,
            "confidence": node.confidence,
            "properties": node.properties,
        },
        "edges": unique_edges,
        "rules": rule_dicts,
        "neighbors": all_neighbors,
    }


def _rules_impl(
    domains: list[str] | None = None,
    concept_ids: list[str] | None = None,
    categories: list[str] | None = None,
    include_derived: bool = True,
    min_confidence: float = 0.0,
) -> dict:
    """Get projected rules from the knowledge graph."""
    _ensure_initialized()

    from qortex.projectors.models import ProjectionFilter
    from qortex.projectors.sources.flat import FlatRuleSource

    filt = ProjectionFilter(
        domains=domains,
        categories=categories,
        min_confidence=min_confidence,
    )

    source = FlatRuleSource(backend=_backend, include_derived=include_derived)
    all_rules = source.derive(domains=domains, filters=filt)

    # Filter by concept_ids if provided
    if concept_ids is not None:
        concept_set = set(concept_ids)
        all_rules = [
            r for r in all_rules
            if concept_set.intersection(r.source_concepts)
        ]

    domain_names = {r.domain for r in all_rules}

    return {
        "rules": [
            {
                "id": r.id,
                "text": r.text,
                "domain": r.domain,
                "category": r.category,
                "confidence": r.confidence,
                "relevance": r.relevance,
                "derivation": r.derivation,
                "source_concepts": r.source_concepts,
                "metadata": r.metadata,
            }
            for r in all_rules
        ],
        "domain_count": len(domain_names),
        "projection": "rules",
    }


# ---------------------------------------------------------------------------
# MCP tool wrappers (thin delegates to _impl functions)
# ---------------------------------------------------------------------------


@mcp.tool
def qortex_query(
    context: str,
    domains: list[str] | None = None,
    top_k: int = 20,
    min_confidence: float = 0.0,
    mode: str = "auto",
) -> dict:
    """Retrieve relevant knowledge items for a context.

    Uses vector similarity search, optionally enhanced with graph-based
    PPR (Personalized PageRank) for deeper structural retrieval.

    Args:
        context: Query text (file content, task description, question).
        domains: Restrict search to these domains. None = search all.
        top_k: Maximum number of items to return.
        min_confidence: Minimum similarity score (0.0 to 1.0).
        mode: Retrieval mode. "vec" = cosine similarity only (Level 0).
            "graph" = PPR over merged graph (Level 1+). "auto" = best available.
    """
    return _query_impl(context, domains, top_k, min_confidence, mode)


@mcp.tool
def qortex_feedback(
    query_id: str,
    outcomes: dict[str, str],
    source: str = "unknown",
) -> dict:
    """Report outcomes for retrieved items to improve future retrieval.

    After using qortex_query results, report which items were useful.
    This data feeds teleportation factor updates (Phase 4).

    Args:
        query_id: The query_id from a qortex_query response.
        outcomes: Mapping of item_id to "accepted", "rejected", or "partial".
        source: Identifier for the consumer reporting feedback.
    """
    return _feedback_impl(query_id, outcomes, source)


@mcp.tool
def qortex_ingest(
    source_path: str,
    domain: str,
    source_type: str | None = None,
) -> dict:
    """Ingest a file into a domain.

    Runs the ingestion pipeline: chunk → LLM extraction → concepts + edges + rules.
    Generates embeddings if qortex[vec] is installed.

    Args:
        source_path: Path to the file to ingest.
        domain: Target domain name (e.g. "memory/default", "buildlog/rules").
        source_type: File type override: "text", "markdown", or "pdf". Auto-detected if None.
    """
    return _ingest_impl(source_path, domain, source_type)


@mcp.tool
def qortex_domains() -> dict:
    """List available domains and their stats."""
    return _domains_impl()


@mcp.tool
def qortex_status() -> dict:
    """Check server health and backend info."""
    return _status_impl()


@mcp.tool
def qortex_explore(
    node_id: str,
    depth: int = 1,
) -> dict:
    """Explore a node's neighborhood in the knowledge graph.

    Returns the node, its typed edges, neighbor nodes, and linked rules.
    Returns {"node": null} if the node doesn't exist.

    Args:
        node_id: The concept node ID to explore.
        depth: How many hops to traverse (1-3). Default 1 = immediate neighbors.
    """
    result = _explore_impl(node_id, depth)
    if result is None:
        return {"node": None}
    return result


@mcp.tool
def qortex_rules(
    domains: list[str] | None = None,
    concept_ids: list[str] | None = None,
    categories: list[str] | None = None,
    include_derived: bool = True,
    min_confidence: float = 0.0,
) -> dict:
    """Get projected rules from the knowledge graph.

    Delegates to the FlatRuleSource projector system. Returns both explicit
    rules (from ingestion) and derived rules (from edge templates).

    Args:
        domains: Filter to these domains. None = all.
        concept_ids: Only return rules linked to these concept IDs.
        categories: Filter by rule category (e.g. "architectural", "testing").
        include_derived: Include edge-derived rules (default True).
        min_confidence: Minimum rule confidence (0.0 to 1.0).
    """
    return _rules_impl(domains, concept_ids, categories, include_derived, min_confidence)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_llm_backend: Any = None


def _get_llm_backend():
    """Get or create LLM backend for ingestion."""
    global _llm_backend

    if _llm_backend is not None:
        return _llm_backend

    # Try to load real LLM backend
    try:
        from qortex_ingest.backends.anthropic import AnthropicBackend

        _llm_backend = AnthropicBackend()
        return _llm_backend
    except (ImportError, Exception):
        pass

    # Fallback to stub
    from qortex_ingest.base import StubLLMBackend

    _llm_backend = StubLLMBackend()
    return _llm_backend


def set_llm_backend(llm) -> None:
    """Set the LLM backend for ingestion. Used by tests and advanced configs."""
    global _llm_backend
    _llm_backend = llm


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def serve(transport: str = "stdio") -> None:
    """Start the qortex MCP server.

    Args:
        transport: "stdio" (default) or "sse".
    """
    _ensure_initialized()
    mcp.run(transport=transport)
