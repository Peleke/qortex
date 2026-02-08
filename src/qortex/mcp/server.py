"""MCP server implementation for qortex.

Tools exposed:
- qortex_query: Retrieve relevant knowledge items for a context
- qortex_feedback: Report outcomes for retrieved items
- qortex_ingest: Ingest a file into a domain
- qortex_domains: List available domains
- qortex_status: Server health + backend info
- qortex_explore: Explore a node's neighborhood in the graph
- qortex_rules: Get projected rules from the knowledge graph
- qortex_source_connect: Connect to a database source
- qortex_source_discover: Discover schemas from a connected source
- qortex_source_sync: Sync source data to vec layer
- qortex_source_list: List connected sources
- qortex_source_disconnect: Disconnect a source
- qortex_vector_create_index: Create a named vector index
- qortex_vector_list_indexes: List named vector indexes
- qortex_vector_describe_index: Get index stats
- qortex_vector_delete_index: Delete a named index
- qortex_vector_upsert: Upsert vectors into an index
- qortex_vector_query: Query vectors by similarity
- qortex_vector_update: Update vector/metadata
- qortex_vector_delete: Delete a single vector
- qortex_vector_delete_many: Delete vectors by IDs or filter

Architecture:
    Each tool has a plain `_<name>_impl()` function with the core logic,
    plus an `@mcp.tool`-decorated wrapper that delegates to it.
    Tests call the `_impl` functions directly; MCP clients hit the wrappers.

    Vec layer and graph layer are independent:
        QORTEX_VEC=memory|sqlite   → VectorIndex (similarity search)
        QORTEX_GRAPH=memory|memgraph → GraphBackend (node metadata, PPR, rules)
    The adapter composes both layers.

    Vector-level tools (qortex_vector_*) manage a separate registry of
    named indexes for MastraVector and other raw-vector consumers.
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
from qortex.sources.registry import SourceRegistry

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

# ---------------------------------------------------------------------------
# Source registry (for database source adapters)
# ---------------------------------------------------------------------------

_source_registry = SourceRegistry()

# ---------------------------------------------------------------------------
# Vector-level index registry (for MastraVector / raw vector operations)
# ---------------------------------------------------------------------------

_vector_indexes: dict[str, Any] = {}  # name -> VectorIndex instance
_index_configs: dict[str, dict] = {}  # name -> {dimension, metric}
_vector_metadata: dict[str, dict[str, dict]] = {}  # name -> {id -> metadata}
_vector_documents: dict[str, dict[str, str]] = {}  # name -> {id -> document}

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
        # Eagerly verify the underlying model is loadable
        _ = _embedding_model.dimensions
    except (ImportError, Exception) as e:
        logger.warning("qortex[vec] not installed (%s). Vector search unavailable.", e)
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
            _vector_index,
            _backend,
            _embedding_model,
            interoception=_interoception,
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
    global _vector_indexes, _index_configs, _vector_metadata, _vector_documents
    global _source_registry

    _backend = backend
    _embedding_model = embedding_model

    # Reset registries (clean state for tests)
    _vector_indexes = {}
    _index_configs = {}
    _vector_metadata = {}
    _vector_documents = {}
    _source_registry = SourceRegistry()

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
            _vector_index,
            _backend,
            _embedding_model,
            interoception=_interoception,
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


_ALLOWED_OUTCOMES = {"accepted", "rejected", "partial"}


def _feedback_impl(
    query_id: str,
    outcomes: dict[str, str],
    source: str = "unknown",
) -> dict:
    _ensure_initialized()

    # Validate outcome values
    for item_id, outcome in outcomes.items():
        if outcome not in _ALLOWED_OUTCOMES:
            return {
                "error": f"Invalid outcome '{outcome}' for item '{item_id}'. "
                f"Must be one of: {', '.join(sorted(_ALLOWED_OUTCOMES))}",
            }

    # Route feedback to both adapters
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
        return {
            "error": f"Invalid source_type: {source_type}. Must be one of {_ALLOWED_SOURCE_TYPES}"
        }

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


def _ingest_text_impl(
    text: str,
    domain: str,
    format: str = "text",
    name: str | None = None,
) -> dict:
    """Ingest raw text or markdown into a domain."""
    _ensure_initialized()

    if format not in ("text", "markdown"):
        return {"error": f"Invalid format: {format}. Must be 'text' or 'markdown'"}

    if not text or not text.strip():
        return {
            "domain": domain,
            "source": name or "raw_text",
            "concepts": 0,
            "edges": 0,
            "rules": 0,
            "warnings": ["Empty text provided"],
        }

    from qortex_ingest.base import Source

    source = Source(
        raw_content=text,
        source_type=format,
        name=name or "raw_text",
    )

    llm = _get_llm_backend()

    if format == "markdown":
        from qortex_ingest.markdown import MarkdownIngestor

        ingestor = MarkdownIngestor(llm, embedding_model=_embedding_model)
    else:
        from qortex_ingest.text import TextIngestor

        ingestor = TextIngestor(llm, embedding_model=_embedding_model)

    manifest = ingestor.ingest(source, domain=domain)
    _backend.ingest_manifest(manifest)

    # Dual-write embeddings
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
        "source": name or "raw_text",
        "concepts": len(manifest.concepts),
        "edges": len(manifest.edges),
        "rules": len(manifest.rules),
        "warnings": manifest.warnings,
    }


def _ingest_structured_impl(
    concepts: list[dict],
    domain: str,
    edges: list[dict] | None = None,
    rules: list[dict] | None = None,
) -> dict:
    """Ingest pre-structured data directly into the knowledge graph."""
    import hashlib

    _ensure_initialized()

    from qortex.core.models import (
        ConceptEdge,
        ConceptNode,
        ExplicitRule,
        IngestionManifest,
        RelationType,
        SourceMetadata,
    )

    edges = edges or []
    rules = rules or []

    if _backend.get_domain(domain) is None:
        _backend.create_domain(domain)

    source_id = f"structured:{hashlib.sha256(domain.encode()).hexdigest()[:12]}"

    # Build ConceptNodes
    concept_nodes: list[ConceptNode] = []
    name_to_id: dict[str, str] = {}
    for c in concepts:
        cname = c.get("name")
        if not cname:
            continue
        desc = c.get("description", cname)
        node_id = c.get("id", f"{domain}:{hashlib.sha256(cname.encode()).hexdigest()[:12]}")
        node = ConceptNode(
            id=node_id,
            name=cname,
            description=desc,
            domain=domain,
            source_id=source_id,
            properties=c.get("properties", {}),
            confidence=c.get("confidence", 1.0),
        )
        concept_nodes.append(node)
        name_to_id[cname] = node_id
        name_to_id[node_id] = node_id

    # Generate embeddings
    if _embedding_model is not None and concept_nodes:
        texts = [f"{n.name}: {n.description}" for n in concept_nodes]
        embs = _embedding_model.embed(texts)
        for node, emb in zip(concept_nodes, embs):
            node.embedding = emb

    # Build ConceptEdges
    concept_edges: list[ConceptEdge] = []
    for e in edges:
        source_ref = e.get("source", "")
        target_ref = e.get("target", "")
        rel_type_str = e.get("relation_type", "")

        source_node_id = name_to_id.get(source_ref)
        target_node_id = name_to_id.get(target_ref)
        if source_node_id is None or target_node_id is None:
            continue

        try:
            rel_type = RelationType(rel_type_str)
        except ValueError:
            continue

        concept_edges.append(
            ConceptEdge(
                source_id=source_node_id,
                target_id=target_node_id,
                relation_type=rel_type,
                confidence=e.get("confidence", 1.0),
            )
        )

    # Build ExplicitRules
    explicit_rules: list[ExplicitRule] = []
    for r in rules:
        rtext = r.get("text", "")
        if not rtext:
            continue
        rule_id = f"{domain}:rule:{hashlib.sha256(rtext.encode()).hexdigest()[:12]}"
        explicit_rules.append(
            ExplicitRule(
                id=rule_id,
                text=rtext,
                domain=domain,
                source_id=source_id,
                category=r.get("category"),
                confidence=r.get("confidence", 1.0),
                concept_ids=[n.id for n in concept_nodes],
            )
        )

    source_meta = SourceMetadata(
        id=source_id,
        name="structured_input",
        source_type="text",
        path_or_url="structured://input",
    )

    manifest = IngestionManifest(
        source=source_meta,
        domain=domain,
        concepts=concept_nodes,
        edges=concept_edges,
        rules=explicit_rules,
    )

    _backend.ingest_manifest(manifest)

    # Dual-write embeddings
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
        "source": "structured_input",
        "concepts": len(concept_nodes),
        "edges": len(concept_edges),
        "rules": len(explicit_rules),
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
                    "relation_type": edge.relation_type.value
                    if hasattr(edge.relation_type, "value")
                    else str(edge.relation_type),
                    "confidence": edge.confidence,
                    "properties": edge.properties,
                }
                all_edges.append(edge_dict)
                neighbor_id = edge.target_id if edge.source_id == current_id else edge.source_id
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor_node = _backend.get_node(neighbor_id)
                    if neighbor_node is not None:
                        all_neighbors.append(
                            {
                                "id": neighbor_node.id,
                                "name": neighbor_node.name,
                                "description": neighbor_node.description,
                                "domain": neighbor_node.domain,
                                "confidence": neighbor_node.confidence,
                                "properties": neighbor_node.properties,
                            }
                        )
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
        all_rules = [r for r in all_rules if concept_set.intersection(r.source_concepts)]

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
# Source-level operations (database source adapters)
# ---------------------------------------------------------------------------


def _source_connect_impl(
    source_id: str,
    connection_string: str,
    schemas: list[str] | None = None,
    domain_map: dict[str, str] | None = None,
    targets: str = "both",
) -> dict:
    """Connect to a database source, discover schemas."""
    import asyncio

    from qortex.sources.base import IngestConfig, SourceConfig

    _ensure_initialized()

    config = SourceConfig(
        source_id=source_id,
        connection_string=connection_string,
        schemas=schemas or ["public"],
        domain_map=domain_map or {},
    )

    try:
        from qortex.sources.postgres import PostgresSourceAdapter
    except ImportError:
        return {"error": "asyncpg not installed. Install qortex[source-postgres]."}

    adapter = PostgresSourceAdapter()

    try:
        asyncio.get_event_loop().run_until_complete(adapter.connect(config))
        table_schemas = asyncio.get_event_loop().run_until_complete(adapter.discover())
    except Exception as e:
        return {"error": f"Connection failed: {e}"}

    _source_registry.register(config, adapter)
    _source_registry.cache_schemas(source_id, table_schemas)

    return {
        "status": "connected",
        "source_id": source_id,
        "tables": len(table_schemas),
        "table_names": [t.name for t in table_schemas],
    }


def _source_discover_impl(source_id: str) -> dict:
    """Return cached schemas for a connected source."""
    schemas = _source_registry.get_schemas(source_id)
    if schemas is None:
        return {"error": f"Source '{source_id}' not found. Connect first."}

    return {
        "source_id": source_id,
        "tables": [
            {
                "name": t.name,
                "schema": t.schema_name,
                "columns": len(t.columns),
                "row_count": t.row_count,
                "pk_columns": t.pk_columns,
                "fk_count": len(t.fk_columns),
            }
            for t in schemas
        ],
    }


def _source_sync_impl(
    source_id: str,
    tables: list[str] | None = None,
    mode: str = "full",
) -> dict:
    """Sync a connected source's data to the vec layer."""
    import asyncio

    _ensure_initialized()

    adapter = _source_registry.get(source_id)
    config = _source_registry.get_config(source_id)
    if adapter is None or config is None:
        return {"error": f"Source '{source_id}' not found. Connect first."}

    try:
        result = asyncio.get_event_loop().run_until_complete(
            adapter.sync(
                tables=tables,
                mode=mode,
                vector_index=_vector_index,
                embedding_model=_embedding_model,
            )
        )
        return {
            "source_id": result.source_id,
            "tables_synced": result.tables_synced,
            "rows_added": result.rows_added,
            "vectors_created": result.vectors_created,
            "duration_seconds": result.duration_seconds,
            "errors": result.errors,
        }
    except Exception as e:
        return {"error": f"Sync failed: {e}"}


def _source_list_impl() -> dict:
    """List connected sources."""
    sources = _source_registry.list_sources()
    return {
        "sources": [
            {
                "source_id": sid,
                "tables": len(_source_registry.get_schemas(sid) or []),
            }
            for sid in sources
        ]
    }


def _source_disconnect_impl(source_id: str) -> dict:
    """Disconnect a source and remove it from the registry."""
    import asyncio

    adapter = _source_registry.get(source_id)
    if adapter is None:
        return {"error": f"Source '{source_id}' not found."}

    try:
        asyncio.get_event_loop().run_until_complete(adapter.disconnect())
    except Exception:
        pass

    _source_registry.remove(source_id)
    return {"status": "disconnected", "source_id": source_id}


# ---------------------------------------------------------------------------
# Vector-level operations (for MastraVector / raw vector consumers)
# ---------------------------------------------------------------------------


def _match_filter(metadata: dict, filt: dict) -> bool:
    """Evaluate a MongoDB-like filter against a metadata dict.

    Supports: equality, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or, $not.
    """
    for key, value in filt.items():
        if key == "$and":
            if not all(_match_filter(metadata, sub) for sub in value):
                return False
        elif key == "$or":
            if not any(_match_filter(metadata, sub) for sub in value):
                return False
        elif key == "$not":
            if _match_filter(metadata, value):
                return False
        elif isinstance(value, dict):
            # Operator expression: {field: {$op: val}}
            actual = metadata.get(key)
            for op, operand in value.items():
                if op == "$ne" and actual == operand:
                    return False
                if op == "$gt" and not (actual is not None and actual > operand):
                    return False
                if op == "$gte" and not (actual is not None and actual >= operand):
                    return False
                if op == "$lt" and not (actual is not None and actual < operand):
                    return False
                if op == "$lte" and not (actual is not None and actual <= operand):
                    return False
                if op == "$in" and actual not in operand:
                    return False
                if op == "$nin" and actual in operand:
                    return False
        else:
            # Simple equality
            if metadata.get(key) != value:
                return False
    return True


def _vector_create_index_impl(
    index_name: str,
    dimension: int,
    metric: str = "cosine",
) -> dict:
    """Create a named vector index."""
    if index_name in _vector_indexes:
        existing = _index_configs[index_name]
        if existing["dimension"] != dimension:
            return {
                "error": f"Index '{index_name}' already exists with dimension "
                f"{existing['dimension']}, requested {dimension}",
            }
        return {"status": "exists", "index_name": index_name}

    try:
        from qortex.vec.index import NumpyVectorIndex
    except ImportError:
        return {
            "error": "numpy is required for vector operations. "
            "Install with: pip install qortex (numpy is a required dependency).",
        }

    _vector_indexes[index_name] = NumpyVectorIndex(dimensions=dimension)
    _index_configs[index_name] = {"dimension": dimension, "metric": metric}
    _vector_metadata[index_name] = {}
    _vector_documents[index_name] = {}

    return {"status": "created", "index_name": index_name}


def _vector_list_indexes_impl() -> dict:
    """List all named vector indexes."""
    return {"indexes": list(_vector_indexes.keys())}


def _vector_describe_index_impl(index_name: str) -> dict:
    """Describe a named vector index."""
    if index_name not in _vector_indexes:
        return {"error": f"Index '{index_name}' not found"}

    idx = _vector_indexes[index_name]
    cfg = _index_configs[index_name]
    return {
        "dimension": cfg["dimension"],
        "count": idx.size(),
        "metric": cfg.get("metric", "cosine"),
    }


def _vector_delete_index_impl(index_name: str) -> dict:
    """Delete a named vector index."""
    if index_name not in _vector_indexes:
        return {"error": f"Index '{index_name}' not found"}

    del _vector_indexes[index_name]
    del _index_configs[index_name]
    _vector_metadata.pop(index_name, None)
    _vector_documents.pop(index_name, None)

    return {"status": "deleted", "index_name": index_name}


def _vector_upsert_impl(
    index_name: str,
    vectors: list[list[float]],
    metadata: list[dict] | None = None,
    ids: list[str] | None = None,
    documents: list[str] | None = None,
) -> dict:
    """Upsert vectors into a named index."""
    if index_name not in _vector_indexes:
        return {"error": f"Index '{index_name}' not found"}

    import uuid as _uuid

    idx = _vector_indexes[index_name]
    cfg = _index_configs[index_name]
    generated_ids = ids or [str(_uuid.uuid4()) for _ in vectors]

    if len(generated_ids) != len(vectors):
        return {"error": f"ids ({len(generated_ids)}) and vectors ({len(vectors)}) must match"}

    # Validate vector dimensions
    expected_dim = cfg["dimension"]
    for i, vec in enumerate(vectors):
        if len(vec) != expected_dim:
            return {
                "error": f"Vector {i} has dimension {len(vec)}, expected {expected_dim}",
            }

    idx.add(generated_ids, vectors)

    # Store metadata alongside vectors
    meta_store = _vector_metadata[index_name]
    doc_store = _vector_documents[index_name]
    meta_list = metadata or [{}] * len(vectors)
    doc_list: list[str | None] = list(documents) if documents is not None else [None] * len(vectors)

    for vid, meta, doc in zip(generated_ids, meta_list, doc_list):
        meta_store[vid] = meta or {}
        if doc is not None:
            doc_store[vid] = doc

    return {"ids": generated_ids}


def _vector_query_impl(
    index_name: str,
    query_vector: list[float],
    top_k: int = 10,
    filter: dict | None = None,
    include_vector: bool = False,
) -> dict:
    """Query vectors from a named index."""
    if index_name not in _vector_indexes:
        return {"error": f"Index '{index_name}' not found"}

    idx = _vector_indexes[index_name]
    cfg = _index_configs[index_name]
    meta_store = _vector_metadata[index_name]
    doc_store = _vector_documents[index_name]

    # Validate query vector dimension
    expected_dim = cfg["dimension"]
    if len(query_vector) != expected_dim:
        return {
            "error": f"Query vector has dimension {len(query_vector)}, expected {expected_dim}",
        }

    # If we have a filter, we need to over-fetch and post-filter
    fetch_k = top_k * 5 if filter else top_k
    raw_results = idx.search(query_vector, top_k=fetch_k)

    results: list[dict] = []
    for vid, score in raw_results:
        meta = meta_store.get(vid, {})
        if filter and not _match_filter(meta, filter):
            continue

        result: dict = {
            "id": vid,
            "score": round(score, 4),
            "metadata": meta,
        }
        if vid in doc_store:
            result["document"] = doc_store[vid]
        if include_vector:
            result["vector"] = None  # VectorIndex doesn't store raw vectors
        results.append(result)

        if len(results) >= top_k:
            break

    return {"results": results}


def _vector_update_impl(
    index_name: str,
    id: str | None = None,
    filter: dict | None = None,
    vector: list[float] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Update a vector's embedding and/or metadata."""
    if index_name not in _vector_indexes:
        return {"error": f"Index '{index_name}' not found"}

    if not id and not filter:
        return {"error": "Either id or filter must be provided"}
    if id and filter:
        return {"error": "Cannot specify both id and filter"}
    if not vector and not metadata:
        return {"error": "No updates provided"}

    idx = _vector_indexes[index_name]
    cfg = _index_configs[index_name]
    meta_store = _vector_metadata[index_name]

    # Validate update vector dimension
    if vector is not None:
        expected_dim = cfg["dimension"]
        if len(vector) != expected_dim:
            return {
                "error": f"Update vector has dimension {len(vector)}, expected {expected_dim}",
            }

    # Resolve target IDs
    if id:
        target_ids = [id]
    else:
        target_ids = [vid for vid, meta in meta_store.items() if _match_filter(meta, filter)]

    updated = 0
    for vid in target_ids:
        if vector is not None:
            idx.add([vid], [vector])
        if metadata is not None:
            existing = meta_store.get(vid, {})
            existing.update(metadata)
            meta_store[vid] = existing
        updated += 1

    return {"status": "updated", "count": updated}


def _vector_delete_impl(index_name: str, id: str) -> dict:
    """Delete a single vector by ID."""
    if index_name not in _vector_indexes:
        return {"error": f"Index '{index_name}' not found"}

    idx = _vector_indexes[index_name]
    idx.remove([id])
    _vector_metadata[index_name].pop(id, None)
    _vector_documents[index_name].pop(id, None)

    return {"status": "deleted", "id": id}


def _vector_delete_many_impl(
    index_name: str,
    ids: list[str] | None = None,
    filter: dict | None = None,
) -> dict:
    """Delete multiple vectors by IDs or filter."""
    if index_name not in _vector_indexes:
        return {"error": f"Index '{index_name}' not found"}

    if not ids and not filter:
        return {"error": "Either ids or filter must be provided"}
    if ids and filter:
        return {"error": "Cannot specify both ids and filter"}

    idx = _vector_indexes[index_name]
    meta_store = _vector_metadata[index_name]
    doc_store = _vector_documents[index_name]

    if filter:
        ids = [vid for vid, meta in meta_store.items() if _match_filter(meta, filter)]

    idx.remove(ids)
    for vid in ids:
        meta_store.pop(vid, None)
        doc_store.pop(vid, None)

    return {"status": "deleted", "count": len(ids)}


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
def qortex_ingest_text(
    text: str,
    domain: str,
    format: str = "text",
    name: str | None = None,
) -> dict:
    """Ingest raw text or markdown into a domain.

    No file needed — paste text directly. Runs the full LLM extraction
    pipeline (concepts + typed edges + rules) on the provided content.

    Args:
        text: The text content to ingest.
        domain: Target domain name.
        format: "text" or "markdown". Selects the chunking strategy.
        name: Optional human-readable source name.
    """
    return _ingest_text_impl(text, domain, format, name)


@mcp.tool
def qortex_ingest_structured(
    concepts: list[dict],
    domain: str,
    edges: list[dict] | None = None,
    rules: list[dict] | None = None,
) -> dict:
    """Ingest pre-structured data directly into the knowledge graph.

    Bypasses LLM extraction — takes concepts, edges, and rules directly.
    Use when you already have structured data to add.

    Args:
        concepts: List of concept dicts, each with "name" and "description".
        domain: Target domain name.
        edges: Optional edges. Each dict: {"source": "name", "target": "name", "relation_type": "requires"}.
        rules: Optional rules. Each dict: {"text": "rule text", "category": "optional"}.
    """
    return _ingest_structured_impl(concepts, domain, edges, rules)


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
# Source-level MCP tools (database source adapters)
# ---------------------------------------------------------------------------


@mcp.tool
def qortex_source_connect(
    source_id: str,
    connection_string: str,
    schemas: list[str] | None = None,
    domain_map: dict[str, str] | None = None,
    targets: str = "both",
) -> dict:
    """Connect to a PostgreSQL database and discover its schema.

    Args:
        source_id: Unique identifier for this source (e.g. "mm_movements").
        connection_string: PostgreSQL connection URL.
        schemas: Database schemas to discover. Default: ["public"].
        domain_map: Glob pattern → domain name mapping.
        targets: Ingestion targets: "vec", "graph", or "both".
    """
    return _source_connect_impl(source_id, connection_string, schemas, domain_map, targets)


@mcp.tool
def qortex_source_discover(source_id: str) -> dict:
    """Return discovered schemas for a connected source.

    Args:
        source_id: The source to describe.
    """
    return _source_discover_impl(source_id)


@mcp.tool
def qortex_source_sync(
    source_id: str,
    tables: list[str] | None = None,
    mode: str = "full",
) -> dict:
    """Sync a database source to the vector layer.

    Serializes rows to text, generates embeddings, and upserts to VectorIndex.

    Args:
        source_id: The source to sync.
        tables: Tables to sync. None = all discovered tables.
        mode: "full" (re-sync everything) or "incremental" (only changes).
    """
    return _source_sync_impl(source_id, tables, mode)


@mcp.tool
def qortex_source_list() -> dict:
    """List all connected database sources."""
    return _source_list_impl()


@mcp.tool
def qortex_source_disconnect(source_id: str) -> dict:
    """Disconnect a database source and remove it from the registry.

    Args:
        source_id: The source to disconnect.
    """
    return _source_disconnect_impl(source_id)


# ---------------------------------------------------------------------------
# Vector-level MCP tools (for MastraVector / raw vector consumers)
# ---------------------------------------------------------------------------


@mcp.tool
def qortex_vector_create_index(
    index_name: str,
    dimension: int,
    metric: str = "cosine",
) -> dict:
    """Create a named vector index for raw vector operations.

    Used by MastraVector and other vector-level consumers. Separate from
    the text-level index used by qortex_query/qortex_ingest.

    Args:
        index_name: Unique name for the index.
        dimension: Vector dimensionality (e.g. 384, 768, 1536).
        metric: Distance metric: "cosine", "euclidean", or "dotproduct".
    """
    return _vector_create_index_impl(index_name, dimension, metric)


@mcp.tool
def qortex_vector_list_indexes() -> dict:
    """List all named vector indexes."""
    return _vector_list_indexes_impl()


@mcp.tool
def qortex_vector_describe_index(index_name: str) -> dict:
    """Get statistics for a named vector index.

    Returns dimension, vector count, and distance metric.

    Args:
        index_name: The index to describe.
    """
    return _vector_describe_index_impl(index_name)


@mcp.tool
def qortex_vector_delete_index(index_name: str) -> dict:
    """Delete a named vector index and all its data.

    Args:
        index_name: The index to delete.
    """
    return _vector_delete_index_impl(index_name)


@mcp.tool
def qortex_vector_upsert(
    index_name: str,
    vectors: list[list[float]],
    metadata: list[dict] | None = None,
    ids: list[str] | None = None,
    documents: list[str] | None = None,
) -> dict:
    """Upsert vectors into a named index.

    If IDs already exist, their vectors and metadata are replaced.
    If no IDs are provided, UUIDs are generated.

    Args:
        index_name: Target index.
        vectors: List of embedding vectors (number[][]).
        metadata: Optional metadata per vector.
        ids: Optional IDs. Auto-generated if omitted.
        documents: Optional document text per vector.
    """
    return _vector_upsert_impl(index_name, vectors, metadata, ids, documents)


@mcp.tool
def qortex_vector_query(
    index_name: str,
    query_vector: list[float],
    top_k: int = 10,
    filter: dict | None = None,
    include_vector: bool = False,
) -> dict:
    """Query a named vector index by similarity.

    Returns results sorted by descending similarity score.
    Supports MongoDB-like metadata filters ($eq, $ne, $gt, $lt, $in, $and, $or).

    Args:
        index_name: Index to query.
        query_vector: Query embedding vector.
        top_k: Maximum results to return.
        filter: Optional metadata filter (MongoDB-like syntax).
        include_vector: Whether to include the vector in results.
    """
    return _vector_query_impl(index_name, query_vector, top_k, filter, include_vector)


@mcp.tool
def qortex_vector_update(
    index_name: str,
    id: str | None = None,
    filter: dict | None = None,
    vector: list[float] | None = None,
    metadata: dict | None = None,
) -> dict:
    """Update a vector's embedding and/or metadata.

    Specify either id (single vector) or filter (multiple vectors).
    At least one of vector or metadata must be provided.

    Args:
        index_name: Target index.
        id: Single vector ID to update. Mutually exclusive with filter.
        filter: Metadata filter for bulk update. Mutually exclusive with id.
        vector: New embedding vector.
        metadata: Metadata fields to merge.
    """
    return _vector_update_impl(index_name, id, filter, vector, metadata)


@mcp.tool
def qortex_vector_delete(index_name: str, id: str) -> dict:
    """Delete a single vector by ID.

    Args:
        index_name: Target index.
        id: Vector ID to delete.
    """
    return _vector_delete_impl(index_name, id)


@mcp.tool
def qortex_vector_delete_many(
    index_name: str,
    ids: list[str] | None = None,
    filter: dict | None = None,
) -> dict:
    """Delete multiple vectors by IDs or metadata filter.

    Specify either ids or filter (mutually exclusive).

    Args:
        index_name: Target index.
        ids: List of vector IDs to delete.
        filter: Metadata filter for bulk delete.
    """
    return _vector_delete_many_impl(index_name, ids, filter)


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
    except ImportError:
        pass
    except Exception as e:
        logger.warning("Failed to initialize AnthropicBackend: %s", e)

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
