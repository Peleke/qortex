"""MCP server implementation for qortex.

Tools exposed (35 total):

Core tools:
- qortex_query: Retrieve relevant knowledge for a question
- qortex_feedback: Report whether retrieved items were useful
- qortex_ingest: Ingest a file into the knowledge graph
- qortex_ingest_text: Ingest raw text into a domain
- qortex_ingest_structured: Ingest structured JSON data
- qortex_ingest_message: Index a session message (lightweight, no LLM)
- qortex_ingest_tool_result: Index a tool result (lightweight, no LLM)
- qortex_domains: List available knowledge domains
- qortex_status: Server health and backend info
- qortex_explore: Explore a node's neighborhood in the graph
- qortex_rules: Get rules from the knowledge graph
- qortex_compare: Compare graph-enhanced vs cosine-only retrieval
- qortex_stats: Knowledge coverage, learning progress, and activity

Source tools:
- qortex_source_connect: Connect to a database source
- qortex_source_discover: Discover schemas from a connected source
- qortex_source_sync: Sync source data to vec layer
- qortex_source_list: List connected sources
- qortex_source_disconnect: Disconnect a source
- qortex_source_inspect_schema: Inspect database schema with constraint metadata
- qortex_source_ingest_graph: Ingest database schema into knowledge graph

Vector tools:
- qortex_vector_create_index: Create a named vector index
- qortex_vector_list_indexes: List named vector indexes
- qortex_vector_describe_index: Get index stats
- qortex_vector_delete_index: Delete a named index
- qortex_vector_upsert: Upsert vectors into an index
- qortex_vector_query: Query vectors by similarity
- qortex_vector_update: Update vector/metadata
- qortex_vector_delete: Delete a single vector
- qortex_vector_delete_many: Delete vectors by IDs or filter

Learning tools:
- qortex_learning_select: Select items using adaptive learning
- qortex_learning_observe: Record outcome for a selected item
- qortex_learning_posteriors: Get posterior distributions
- qortex_learning_metrics: Get aggregate learning metrics
- qortex_learning_session_start: Start a learning session
- qortex_learning_session_end: End a learning session
- qortex_learning_reset: Reset (delete) learned posteriors for a learner

Architecture:
    Each tool has a plain `_<name>_impl()` function with the core logic,
    plus an `@mcp.tool`-decorated wrapper that delegates to it.
    Tests call the `_impl` functions directly; MCP clients hit the wrappers.

    Vec layer and graph layer are independent:
        QORTEX_VEC=memory|sqlite   -> VectorIndex (similarity search)
        QORTEX_GRAPH=memory|memgraph -> GraphBackend (node metadata, PPR, rules)
    The adapter composes both layers.

    Vector-level tools (qortex_vector_*) manage a separate registry of
    named indexes for MastraVector and other raw-vector consumers.
"""

from __future__ import annotations

import atexit
import functools
import os
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from qortex.core.memory import InMemoryBackend
from qortex.hippocampus.adapter import VecOnlyAdapter
from qortex.observe.logging import get_logger
from qortex.observe.mcp import mcp_trace_middleware
from qortex.sources.registry import SourceRegistry

logger = get_logger(__name__)


def _mcp_traced(fn):
    """Wrap an MCP tool handler with distributed trace middleware."""

    @functools.wraps(fn)
    def wrapper(**kwargs):
        return mcp_trace_middleware(fn.__name__, kwargs, lambda p: fn(**p))

    return wrapper


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
# Learning layer (bandit-based learning)
# ---------------------------------------------------------------------------

_learners: dict[str, Any] = {}  # name -> Learner instance
_learning_state_dir: str = ""  # override for tests (empty = default ~/.qortex/learning)

# ---------------------------------------------------------------------------
# Activity counters (feed qortex_stats)
# Note: not thread-safe. MCP stdio transport is single-threaded, so this is fine.
# ---------------------------------------------------------------------------

_query_count: int = 0
_feedback_count: int = 0
_feedback_outcomes: dict[str, int] = {"accepted": 0, "rejected": 0, "partial": 0}

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
            logger.warning("interoception.shutdown.failed", exc_info=True)


def _ensure_initialized() -> None:
    """Lazy initialization: set up vec + graph layers from env config."""
    global _backend, _vector_index, _adapter, _embedding_model, _interoception

    if _backend is not None:
        return

    # Initialize observability (env-var driven, zero-config by default)
    from qortex.observe import configure

    configure()

    # --- Embedding model ---
    try:
        from qortex.vec.embeddings import SentenceTransformerEmbedding

        _embedding_model = SentenceTransformerEmbedding()
        # Eagerly verify the underlying model is loadable
        _ = _embedding_model.dimensions
    except (ImportError, Exception) as e:
        logger.warning("vec.unavailable", error=str(e))
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
                logger.warning("sqlite-vec.unavailable", fallback="NumpyVectorIndex")
                from qortex.vec.index import NumpyVectorIndex

                _vector_index = NumpyVectorIndex(dimensions=_embedding_model.dimensions)
        else:  # "memory"
            from qortex.vec.index import NumpyVectorIndex

            _vector_index = NumpyVectorIndex(dimensions=_embedding_model.dimensions)

    # --- Graph layer (independent of vec) ---
    graph_backend = os.environ.get("QORTEX_GRAPH", "memory")
    if graph_backend == "memgraph":
        try:
            from qortex.core.backend import MemgraphBackend, MemgraphCredentials

            host = os.environ.get("MEMGRAPH_HOST", "localhost")
            port = int(os.environ.get("MEMGRAPH_PORT", "7687"))
            user = os.environ.get("MEMGRAPH_USER", "")
            password = os.environ.get("MEMGRAPH_PASSWORD", "")
            creds = MemgraphCredentials(user=user, password=password) if user else None
            _backend = MemgraphBackend(host=host, port=port, credentials=creds)
            _backend.connect()
        except (ImportError, Exception) as e:
            logger.warning("memgraph.unavailable", error=str(e), fallback="InMemoryBackend")
            _backend = InMemoryBackend(vector_index=_vector_index)
            _backend.connect()
    else:  # "memory"
        _backend = InMemoryBackend(vector_index=_vector_index)
        _backend.connect()

    # --- Interoception layer (feedback lifecycle) ---
    from qortex.hippocampus.interoception import InteroceptionConfig, LocalInteroceptionProvider

    qortex_dir = Path("~/.qortex").expanduser()
    interoception_config = InteroceptionConfig(
        db_path=qortex_dir / "interoception.db",
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
    global _source_registry, _learners

    _backend = backend
    _embedding_model = embedding_model

    # Reset registries (clean state for tests)
    _vector_indexes = {}
    _index_configs = {}
    _vector_metadata = {}
    _vector_documents = {}
    _source_registry = SourceRegistry()
    _learners = {}

    global _query_count, _feedback_count, _feedback_outcomes
    _query_count = 0
    _feedback_count = 0
    _feedback_outcomes = {"accepted": 0, "rejected": 0, "partial": 0}

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

    global _query_count
    _query_count += 1

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

_OUTCOME_REWARD = {"accepted": 1.0, "rejected": -1.0, "partial": 0.3}


def _maybe_propagate_credit(
    query_id: str,
    outcomes: dict[str, str],
) -> dict | None:
    """Propagate credit through causal DAG if flag is enabled.

    Returns summary dict or None if disabled/unavailable.
    """
    from qortex.flags import get_flags

    if not get_flags().credit_propagation:
        return None

    try:
        from qortex.causal.credit import CreditAssigner
        from qortex.causal.dag import CausalDAG
    except ImportError:
        return None

    if not outcomes or _backend is None:
        return None

    # Group concept IDs by domain
    domains: dict[str, list[str]] = {}
    for item_id in outcomes:
        node = _backend.get_node(item_id)
        if node is not None:
            domains.setdefault(node.domain, []).append(item_id)

    if not domains:
        return None

    all_assignments = []
    for domain, concept_ids in domains.items():
        try:
            dag = CausalDAG.from_backend(_backend, domain)
        except Exception:
            continue

        # Compute average reward for this domain's outcomes
        rewards = [_OUTCOME_REWARD.get(outcomes[cid], 0.0) for cid in concept_ids]
        avg_reward = sum(rewards) / len(rewards)

        assigner = CreditAssigner(dag)
        assignments = assigner.assign_credit(concept_ids, avg_reward)
        all_assignments.extend(assignments)

    if not all_assignments:
        return None

    # Convert to posterior updates and apply
    updates = CreditAssigner.to_posterior_updates(all_assignments)
    learner = _get_or_create_learner("credit")
    learner.apply_credit_deltas(updates)

    # Emit event
    from qortex.observe import emit
    from qortex.observe.events import CreditPropagated

    direct = sum(1 for a in all_assignments if a.method == "direct")
    ancestor = sum(1 for a in all_assignments if a.method == "ancestor")
    total_alpha = sum(u.get("alpha_delta", 0.0) for u in updates.values())
    total_beta = sum(u.get("beta_delta", 0.0) for u in updates.values())

    emit(
        CreditPropagated(
            query_id=query_id,
            concept_count=len(updates),
            direct_count=direct,
            ancestor_count=ancestor,
            total_alpha_delta=total_alpha,
            total_beta_delta=total_beta,
            learner="credit",
        )
    )

    return {
        "concept_count": len(updates),
        "direct_count": direct,
        "ancestor_count": ancestor,
    }


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

    global _feedback_count, _feedback_outcomes
    _feedback_count += 1
    for outcome in outcomes.values():
        if outcome in _feedback_outcomes:
            _feedback_outcomes[outcome] += 1

    # Credit propagation (if enabled)
    credit_summary = _maybe_propagate_credit(query_id, outcomes)

    result: dict = {
        "status": "recorded",
        "query_id": query_id,
        "outcome_count": len(outcomes),
        "source": source,
    }
    if credit_summary is not None:
        result["credit"] = credit_summary

    return result


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

    from qortex.ingest.base import Source

    source = Source(
        path=path,
        source_type=source_type,
        name=path.name,
    )

    llm = _get_llm_backend()

    if source_type == "markdown":
        from qortex.ingest.markdown import MarkdownIngestor

        ingestor = MarkdownIngestor(llm, embedding_model=_embedding_model)
    elif source_type == "pdf":
        from qortex.ingest.pdf import PDFIngestor

        ingestor = PDFIngestor(llm, embedding_model=_embedding_model)
    else:
        from qortex.ingest.text import TextIngestor

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

    from qortex.ingest.base import Source

    source = Source(
        raw_content=text,
        source_type=format,
        name=name or "raw_text",
    )

    llm = _get_llm_backend()

    if format == "markdown":
        from qortex.ingest.markdown import MarkdownIngestor

        ingestor = MarkdownIngestor(llm, embedding_model=_embedding_model)
    else:
        from qortex.ingest.text import TextIngestor

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


def _compare_impl(
    context: str,
    domains: list[str] | None = None,
    top_k: int = 5,
) -> dict:
    _ensure_initialized()
    top_k = max(1, min(top_k, 20))

    # Vec-only (flat cosine)
    if _adapter is None:
        return {"error": "No vector index available. Install qortex[vec]."}
    vec_result = _adapter.retrieve(
        query=context,
        domains=domains,
        top_k=top_k,
        min_confidence=0.0,
    )

    # Graph-enhanced (structural + vector)
    graph_result = None
    if _graph_adapter is not None:
        graph_result = _graph_adapter.retrieve(
            query=context,
            domains=domains,
            top_k=top_k,
            min_confidence=0.0,
        )

    def _fmt(result):
        return [
            {
                "rank": i + 1,
                "id": it.id,
                "content": it.content[:200],
                "score": round(it.score, 4),
                "domain": it.domain,
                "node_id": it.node_id,
            }
            for i, it in enumerate(result.items)
        ]

    vec_items = _fmt(vec_result)
    graph_items = _fmt(graph_result) if graph_result else []

    vec_ids = [it["id"] for it in vec_items]
    graph_ids = [it["id"] for it in graph_items]
    vec_set, graph_set = set(vec_ids), set(graph_ids)

    graph_unique = [it for it in graph_items if it["id"] not in vec_set]
    vec_unique = [it for it in vec_items if it["id"] not in graph_set]

    rank_changes = []
    for it in graph_items:
        if it["id"] in vec_set:
            vec_rank = vec_ids.index(it["id"]) + 1
            if vec_rank != it["rank"]:
                rank_changes.append(
                    {
                        "id": it["id"],
                        "content": it["content"][:80],
                        "vec_rank": vec_rank,
                        "graph_rank": it["rank"],
                        "delta": vec_rank - it["rank"],
                    }
                )

    rules = _collect_query_rules(graph_items, domains) if graph_result else []

    return {
        "query": context,
        "vec_only": {"method": "Cosine similarity", "items": vec_items},
        "graph_enhanced": {
            "method": "Graph-enhanced (structural + vector + rules)",
            "items": graph_items,
            "rules_surfaced": len(rules),
            "rules": rules[:3],
        },
        "diff": {
            "graph_found_that_cosine_missed": graph_unique,
            "cosine_found_that_graph_dropped": vec_unique,
            "rank_changes": rank_changes,
            "overlap": len(vec_set & graph_set),
        },
        "summary": _compare_summary(vec_items, graph_items, graph_unique, rules),
    }


def _compare_summary(vec_items, graph_items, graph_unique, rules) -> str:
    if not graph_items:
        return "Graph adapter not available. Showing vector-only results."
    parts = []
    if graph_unique:
        parts.append(f"found {len(graph_unique)} item(s) that cosine missed")
    if rules:
        parts.append(f"surfaced {len(rules)} rule(s)")
    overlap = len(set(i["id"] for i in vec_items) & set(i["id"] for i in graph_items))
    if overlap < len(vec_items):
        parts.append(f"replaced {len(vec_items) - overlap} distractor(s)")
    if not parts:
        return "Both methods returned similar results for this query."
    return "Graph-enhanced retrieval " + ", ".join(parts) + "."


def _stats_impl() -> dict:
    _ensure_initialized()

    # Knowledge coverage
    domains = _backend.list_domains() if _backend else []
    domain_breakdown = {}
    totals = {"concepts": 0, "edges": 0, "rules": 0}
    for d in domains:
        stats = {
            "concepts": d.concept_count,
            "edges": d.edge_count,
            "rules": d.rule_count,
        }
        domain_breakdown[d.name] = stats
        for k in totals:
            totals[k] += stats[k]

    # Learning progress
    learner_breakdown = {}
    total_observations = 0
    for name, lrn in _learners.items():
        m = lrn.metrics()
        learner_breakdown[name] = {
            "total_pulls": m.get("total_pulls", 0),
            "accuracy": round(m.get("accuracy", 0.0), 3),
            "arms_tracked": m.get("arm_count", 0),
            "exploration_ratio": round(m.get("explore_ratio", 0.0), 3),
        }
        total_observations += m.get("total_pulls", 0)

    feedback_rate = round(_feedback_count / _query_count, 3) if _query_count > 0 else 0.0

    return {
        "knowledge": {
            "domains": len(domains),
            **totals,
            "domain_breakdown": domain_breakdown,
        },
        "learning": {
            "learners": len(_learners),
            "total_observations": total_observations,
            "learner_breakdown": learner_breakdown,
        },
        "activity": {
            "queries_served": _query_count,
            "feedback_given": _feedback_count,
            "feedback_rate": feedback_rate,
            "outcomes": dict(_feedback_outcomes),
        },
        "health": {
            "backend": type(_backend).__name__ if _backend else None,
            "vector_index": type(_vector_index).__name__ if _vector_index else None,
            "embedding_model": (
                getattr(_embedding_model, "_model_name", type(_embedding_model).__name__)
                if _embedding_model
                else None
            ),
            "persistence": _get_persistence_info(),
        },
    }


def _get_persistence_info() -> dict:
    state_dir = os.environ.get("QORTEX_STATE_DIR")
    if not state_dir:
        return {"mode": "in-memory", "persistent": False}
    state_path = Path(state_dir)
    if not state_path.exists():
        return {"mode": "configured", "path": str(state_path), "persistent": False}
    files = list(state_path.glob("*.db"))
    size_bytes = sum(f.stat().st_size for f in files if f.is_file())
    return {
        "mode": "sqlite",
        "path": str(state_path),
        "persistent": True,
        "db_files": len(files),
        "size_mb": round(size_bytes / (1024 * 1024), 2),
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

    from qortex.sources.base import SourceConfig

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

    asyncio.get_event_loop().run_until_complete(_source_registry.remove_async(source_id))
    return {"status": "disconnected", "source_id": source_id}


# ---------------------------------------------------------------------------
# Source graph tools (PostgresGraphIngestor — database schema → knowledge graph)
# ---------------------------------------------------------------------------


def _source_inspect_schema_impl(
    connection_string: str,
    source_id: str,
    schemas: list[str] | None = None,
    domain_map: dict[str, str] | None = None,
) -> dict:
    """Inspect a database schema with FK, CHECK, UNIQUE constraint metadata.

    Connects via asyncpg, discovers schema, returns structured overview.
    """
    import asyncio

    try:
        import asyncpg  # noqa: F811
    except ImportError:
        return {"error": "asyncpg not installed. Install qortex[source-postgres]."}

    async def _inspect():
        from qortex.sources.postgres_graph import PostgresGraphIngestor

        conn = await asyncpg.connect(connection_string)
        try:
            ingestor = PostgresGraphIngestor()
            schema = await ingestor.discover_schema(
                conn, source_id=source_id, database_name=source_id
            )
            mapping = ingestor.map_schema(schema, domain_map=domain_map)

            tables_info = []
            table_map = {tm.table_name: tm for tm in mapping.tables}
            for t in schema.tables:
                tm = table_map.get(t.name)
                tables_info.append(
                    {
                        "name": t.name,
                        "schema": t.schema_name,
                        "columns": len(t.columns),
                        "row_count": t.row_count,
                        "foreign_keys": len(t.foreign_keys),
                        "check_constraints": len(t.check_constraints),
                        "unique_constraints": len(t.unique_constraints),
                        "is_catalog": tm.is_catalog if tm else False,
                        "domain": tm.domain if tm else None,
                        "name_column": tm.name_column if tm else None,
                    }
                )

            edges_info = [
                {
                    "source": em.source_table,
                    "target": em.target_table,
                    "fk_column": em.fk_column,
                    "relation_type": em.relation_type,
                }
                for em in mapping.edges
            ]

            rules_info = [
                {
                    "table": rm.table_name,
                    "constraint": rm.constraint_name,
                    "rule_text": rm.rule_text,
                    "category": rm.category,
                }
                for rm in mapping.rules
            ]

            return {
                "source_id": source_id,
                "tables": tables_info,
                "edges": edges_info,
                "rules": rules_info,
                "table_count": len(schema.tables),
                "edge_count": len(mapping.edges),
                "rule_count": len(mapping.rules),
            }
        finally:
            await conn.close()

    try:
        return asyncio.run(_inspect())
    except Exception as e:
        return {"error": str(e)}


def _source_ingest_graph_impl(
    connection_string: str,
    source_id: str,
    schemas: list[str] | None = None,
    domain_map: dict[str, str] | None = None,
    embed_catalog_tables: bool = True,
    extract_rules: bool = True,
) -> dict:
    """Ingest database schema into the knowledge graph.

    Connects via asyncpg, discovers schema, maps to graph structure,
    and ingests ConceptNodes/Edges/Rules into the backend.
    """
    _ensure_initialized()

    import asyncio

    try:
        import asyncpg  # noqa: F811
    except ImportError:
        return {"error": "asyncpg not installed. Install qortex[source-postgres]."}

    async def _ingest():
        from dataclasses import dataclass as _dc

        @_dc
        class _IngestConf:
            embed_catalog_tables: bool = embed_catalog_tables
            extract_rules: bool = extract_rules

        @_dc
        class _SourceConf:
            source_id: str = source_id
            schemas: list[str] = schemas or ["public"]
            domain_map: dict[str, str] = domain_map or {}

        from qortex.sources.postgres_graph import PostgresGraphIngestor

        conn = await asyncpg.connect(connection_string)
        try:
            ingestor = PostgresGraphIngestor(
                config=_SourceConf(),
                ingest_config=_IngestConf(),
                backend=_backend,
                embedding_model=_embedding_model,
            )
            counts = await ingestor.run(conn=conn)
            return {
                "source_id": source_id,
                "concepts": counts.get("concepts", 0),
                "edges": counts.get("edges", 0),
                "rules": counts.get("rules", 0),
            }
        finally:
            await conn.close()

    try:
        return asyncio.run(_ingest())
    except Exception as e:
        return {"error": str(e)}


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
@_mcp_traced
def qortex_query(
    context: str,
    domains: list[str] | None = None,
    top_k: int = 20,
    min_confidence: float = 0.0,
    mode: str = "auto",
) -> dict:
    """Search the project's knowledge graph for context relevant to the current task.

    Call this BEFORE answering questions about:
    - Code architecture, cross-cutting concerns, or component relationships
    - Domain rules, constraints, or best practices
    - How modules interact or depend on each other

    Returns ranked items with content, relevance scores, and applicable rules.
    After using results, call qortex_feedback to report which items helped.

    Args:
        context: What you need context for. Be specific: "How does auth
            middleware validate JWT tokens?" beats "auth".
        domains: Restrict to specific knowledge domains. None = search all.
        top_k: Maximum items to return (default 20).
        min_confidence: Minimum relevance score, 0.0-1.0 (default 0.0).
        mode: "auto" (recommended), "vec" (similarity only), "graph" (structural).
    """
    return _query_impl(context, domains, top_k, min_confidence, mode)


@mcp.tool
@_mcp_traced
def qortex_feedback(
    query_id: str,
    outcomes: dict[str, str],
    source: str = "unknown",
) -> dict:
    """Report which knowledge items were useful after answering a question.

    Call this AFTER using qortex_query results. Items marked "accepted"
    get boosted in future retrievals; "rejected" get suppressed. This is
    how the system gets smarter over time. Always call this.

    Args:
        query_id: The query_id from the qortex_query response.
        outcomes: Map each item_id to "accepted", "rejected", or "partial".
            Only include items you actually used or considered.
        source: Your identifier (e.g. "claude-code", "cursor").
    """
    return _feedback_impl(query_id, outcomes, source)


@mcp.tool
@_mcp_traced
def qortex_ingest(
    source_path: str,
    domain: str,
    source_type: str | None = None,
) -> dict:
    """Add a file to the knowledge graph.

    Extracts concepts, relationships, and rules automatically via LLM
    analysis. Call this when the user wants to teach the system about
    new material: documentation, specs, architecture docs, or code.

    Args:
        source_path: Path to the file to ingest.
        domain: Knowledge domain name (e.g. "auth", "billing", "docs").
        source_type: "text", "markdown", or "pdf". Auto-detected if None.
    """
    return _ingest_impl(source_path, domain, source_type)


@mcp.tool
@_mcp_traced
def qortex_ingest_text(
    text: str,
    domain: str,
    format: str = "text",
    name: str | None = None,
) -> dict:
    """Ingest raw text or markdown into a domain.

    No file needed. Paste text directly. Runs the full LLM extraction
    pipeline (concepts + typed edges + rules) on the provided content.

    Args:
        text: The text content to ingest.
        domain: Target domain name.
        format: "text" or "markdown". Selects the chunking strategy.
        name: Optional human-readable source name.
    """
    return _ingest_text_impl(text, domain, format, name)


@mcp.tool
@_mcp_traced
def qortex_ingest_structured(
    concepts: list[dict],
    domain: str,
    edges: list[dict] | None = None,
    rules: list[dict] | None = None,
) -> dict:
    """Ingest pre-structured data directly into the knowledge graph.

    Bypasses LLM extraction. Takes concepts, edges, and rules directly.
    Use when you already have structured data to add.

    Args:
        concepts: List of concept dicts, each with "name" and "description".
        domain: Target domain name.
        edges: Optional edges. Each dict: {"source": "name", "target": "name", "relation_type": "requires"}.
        rules: Optional rules. Each dict: {"text": "rule text", "category": "optional"}.
    """
    return _ingest_structured_impl(concepts, domain, edges, rules)


@mcp.tool
@_mcp_traced
def qortex_domains() -> dict:
    """List all knowledge domains and their sizes.

    Call this first to discover what knowledge is available before querying.
    """
    return _domains_impl()


@mcp.tool
@_mcp_traced
def qortex_status() -> dict:
    """Check if the knowledge system is healthy and what capabilities are active.

    Call this if queries return unexpected results or errors.
    """
    return _status_impl()


@mcp.tool
@_mcp_traced
def qortex_explore(
    node_id: str,
    depth: int = 1,
) -> dict:
    """Traverse the knowledge graph from a concept to discover connections.

    Call this when you need to understand HOW things connect, not just
    WHAT is relevant (use qortex_query for that). Good for:
    - Tracing dependencies: "what depends on AuthMiddleware?"
    - Understanding hierarchies: "sub-components of PaymentService?"
    - Finding related rules: "what constraints apply to this module?"

    Returns the node, typed edges, neighbors, and linked rules.

    Args:
        node_id: Concept ID to explore (from qortex_query results).
        depth: Hops to traverse (1-3). Start with 1, increase if needed.
    """
    result = _explore_impl(node_id, depth)
    if result is None:
        return {"node": None}
    return result


@mcp.tool
@_mcp_traced
def qortex_rules(
    domains: list[str] | None = None,
    concept_ids: list[str] | None = None,
    categories: list[str] | None = None,
    include_derived: bool = True,
    min_confidence: float = 0.0,
) -> dict:
    """Get domain rules, constraints, and best practices.

    Call this when the user asks about standards, conventions, or
    constraints. Returns explicit rules and patterns derived from the
    graph structure, with confidence scores and linked concepts.

    Args:
        domains: Filter to these domains. None = all.
        concept_ids: Only rules linked to these concepts.
        categories: Filter by category (e.g. "security", "testing").
        include_derived: Include pattern-derived rules (default True).
        min_confidence: Minimum rule confidence, 0.0-1.0.
    """
    return _rules_impl(domains, concept_ids, categories, include_derived, min_confidence)


@mcp.tool
@_mcp_traced
def qortex_compare(
    context: str,
    domains: list[str] | None = None,
    top_k: int = 5,
) -> dict:
    """Compare graph-enhanced vs vanilla retrieval side-by-side.

    Runs the same query through both methods and shows exactly what the
    knowledge graph adds. Use this to demonstrate the value of structural
    retrieval over flat similarity search on your own data.

    Args:
        context: Query to compare both methods on.
        domains: Restrict to these domains. None = search all.
        top_k: Items per method (default 5 for readable comparison).
    """
    return _compare_impl(context, domains, top_k)


@mcp.tool
@_mcp_traced
def qortex_stats() -> dict:
    """See how the knowledge system has improved over time.

    Shows knowledge coverage, learning progress, query activity, and
    system health. Call this to understand the value qortex is providing.
    """
    return _stats_impl()


# ---------------------------------------------------------------------------
# Online indexing tools (session messages + tool results)
# ---------------------------------------------------------------------------


def _ingest_message_impl(
    text: str,
    session_id: str,
    role: str = "user",
    domain: str = "session",
) -> dict:
    """Chunk, embed, and index a session message into the vector layer."""
    _ensure_initialized()
    import time

    if not text or not text.strip():
        return {"session_id": session_id, "chunks": 0, "concepts": 0, "edges": 0}

    start = time.monotonic()

    from qortex.online.chunker import chunk_text

    chunks = chunk_text(text, source_id=f"{session_id}:{role}")

    concepts_added = 0
    edges_added = 0

    if _embedding_model is not None and _vector_index is not None:
        ids = [f"{session_id}:{c.id}" for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = _embedding_model.embed_batch(texts)
        _vector_index.add(ids, embeddings)
        concepts_added = len(ids)

        # Co-occurrence edges between consecutive chunks
        if _backend is not None and len(ids) > 1:
            for i in range(len(ids) - 1):
                _backend.add_edge(ids[i], ids[i + 1], "co_occurs", domain=domain)
                edges_added += 1

    latency_ms = (time.monotonic() - start) * 1000

    from qortex.observe.emitter import emit
    from qortex.observe.events import MessageIngested

    emit(MessageIngested(
        session_id=session_id,
        role=role,
        domain=domain,
        chunk_count=len(chunks),
        concept_count=concepts_added,
        edge_count=edges_added,
        latency_ms=latency_ms,
    ))

    return {
        "session_id": session_id,
        "chunks": len(chunks),
        "concepts": concepts_added,
        "edges": edges_added,
        "latency_ms": round(latency_ms, 2),
    }


def _ingest_tool_result_impl(
    tool_name: str,
    result_text: str,
    session_id: str,
    domain: str = "session",
) -> dict:
    """Chunk, embed, and index a tool result into the vector layer."""
    _ensure_initialized()
    import time

    if not result_text or not result_text.strip():
        return {"tool_name": tool_name, "session_id": session_id, "concepts": 0, "edges": 0}

    start = time.monotonic()

    from qortex.online.chunker import chunk_text

    chunks = chunk_text(result_text, source_id=f"{session_id}:tool:{tool_name}")

    concepts_added = 0
    edges_added = 0

    if _embedding_model is not None and _vector_index is not None:
        ids = [f"{session_id}:tool:{tool_name}:{c.id}" for c in chunks]
        texts = [c.text for c in chunks]
        embeddings = _embedding_model.embed_batch(texts)
        _vector_index.add(ids, embeddings)
        concepts_added = len(ids)

        if _backend is not None and len(ids) > 1:
            for i in range(len(ids) - 1):
                _backend.add_edge(ids[i], ids[i + 1], "co_occurs", domain=domain)
                edges_added += 1

    latency_ms = (time.monotonic() - start) * 1000

    from qortex.observe.emitter import emit
    from qortex.observe.events import ToolResultIngested

    emit(ToolResultIngested(
        tool_name=tool_name,
        session_id=session_id,
        domain=domain,
        concept_count=concepts_added,
        edge_count=edges_added,
        latency_ms=latency_ms,
    ))

    return {
        "tool_name": tool_name,
        "session_id": session_id,
        "concepts": concepts_added,
        "edges": edges_added,
        "latency_ms": round(latency_ms, 2),
    }


@mcp.tool
@_mcp_traced
def qortex_ingest_message(
    text: str,
    session_id: str,
    role: str = "user",
    domain: str = "session",
) -> dict:
    """Index a session message into the vector layer for retrieval.

    Lightweight: chunks text, embeds, adds to vec index. No LLM needed.
    Creates co-occurrence edges between consecutive chunks.

    Args:
        text: The message content.
        session_id: Session identifier for grouping.
        role: Message role ("user", "assistant", "system", "tool").
        domain: Knowledge domain (default "session").
    """
    return _ingest_message_impl(text, session_id, role, domain)


@mcp.tool
@_mcp_traced
def qortex_ingest_tool_result(
    tool_name: str,
    result_text: str,
    session_id: str,
    domain: str = "session",
) -> dict:
    """Index a tool's output into the vector layer for retrieval.

    Lightweight: chunks text, embeds, adds to vec index. No LLM needed.
    Useful for making tool outputs searchable in future queries.

    Args:
        tool_name: Name of the tool that produced the result.
        result_text: The tool's output text.
        session_id: Session identifier for grouping.
        domain: Knowledge domain (default "session").
    """
    return _ingest_tool_result_impl(tool_name, result_text, session_id, domain)


# ---------------------------------------------------------------------------
# Source-level MCP tools (database source adapters + graph ingest)
# ---------------------------------------------------------------------------


@mcp.tool
@_mcp_traced
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
@_mcp_traced
def qortex_source_discover(source_id: str) -> dict:
    """Return discovered schemas for a connected source.

    Args:
        source_id: The source to describe.
    """
    return _source_discover_impl(source_id)


@mcp.tool
@_mcp_traced
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
@_mcp_traced
def qortex_source_list() -> dict:
    """List all connected database sources."""
    return _source_list_impl()


@mcp.tool
@_mcp_traced
def qortex_source_disconnect(source_id: str) -> dict:
    """Disconnect a database source and remove it from the registry.

    Args:
        source_id: The source to disconnect.
    """
    return _source_disconnect_impl(source_id)


@mcp.tool
@_mcp_traced
def qortex_source_inspect_schema(
    connection_string: str,
    source_id: str,
    schemas: list[str] | None = None,
    domain_map: dict[str, str] | None = None,
) -> dict:
    """Inspect a PostgreSQL database schema with full constraint metadata.

    Discovers tables, foreign keys, CHECK constraints, UNIQUE constraints.
    Maps FK relationships to graph edge types and CHECK constraints to rules.
    Returns a structured overview without modifying any data.

    Args:
        connection_string: PostgreSQL connection URL.
        source_id: Identifier for this source (e.g. "mm_movements").
        schemas: Database schemas to inspect. Defaults to ["public"].
        domain_map: Optional glob→domain mapping (e.g. {"habit_*": "habits"}).
    """
    return _source_inspect_schema_impl(connection_string, source_id, schemas, domain_map)


@mcp.tool
@_mcp_traced
def qortex_source_ingest_graph(
    connection_string: str,
    source_id: str,
    schemas: list[str] | None = None,
    domain_map: dict[str, str] | None = None,
    embed_catalog_tables: bool = True,
    extract_rules: bool = True,
) -> dict:
    """Ingest a PostgreSQL database schema into the knowledge graph.

    Discovers schema, classifies FK relationships, detects catalog tables,
    and creates ConceptNodes, ConceptEdges, and ExplicitRules in the backend.

    Args:
        connection_string: PostgreSQL connection URL.
        source_id: Identifier for this source.
        schemas: Database schemas to ingest. Defaults to ["public"].
        domain_map: Optional glob→domain mapping.
        embed_catalog_tables: Embed catalog table rows for similarity search.
        extract_rules: Extract CHECK constraints as ExplicitRules.
    """
    return _source_ingest_graph_impl(
        connection_string,
        source_id,
        schemas,
        domain_map,
        embed_catalog_tables,
        extract_rules,
    )


# ---------------------------------------------------------------------------
# Vector-level MCP tools (for MastraVector / raw vector consumers)
# ---------------------------------------------------------------------------


@mcp.tool
@_mcp_traced
def qortex_vector_create_index(
    index_name: str,
    dimension: int,
    metric: str = "cosine",
) -> dict:
    """Create a named vector index for raw vector operations.

    Used by MastraVector and other vector-level consumers. Separate from
    the text-level index used by qortex_query/qortex.ingest.

    Args:
        index_name: Unique name for the index.
        dimension: Vector dimensionality (e.g. 384, 768, 1536).
        metric: Distance metric: "cosine", "euclidean", or "dotproduct".
    """
    return _vector_create_index_impl(index_name, dimension, metric)


@mcp.tool
@_mcp_traced
def qortex_vector_list_indexes() -> dict:
    """List all named vector indexes."""
    return _vector_list_indexes_impl()


@mcp.tool
@_mcp_traced
def qortex_vector_describe_index(index_name: str) -> dict:
    """Get statistics for a named vector index.

    Returns dimension, vector count, and distance metric.

    Args:
        index_name: The index to describe.
    """
    return _vector_describe_index_impl(index_name)


@mcp.tool
@_mcp_traced
def qortex_vector_delete_index(index_name: str) -> dict:
    """Delete a named vector index and all its data.

    Args:
        index_name: The index to delete.
    """
    return _vector_delete_index_impl(index_name)


@mcp.tool
@_mcp_traced
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
@_mcp_traced
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
@_mcp_traced
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
@_mcp_traced
def qortex_vector_delete(index_name: str, id: str) -> dict:
    """Delete a single vector by ID.

    Args:
        index_name: Target index.
        id: Vector ID to delete.
    """
    return _vector_delete_impl(index_name, id)


@mcp.tool
@_mcp_traced
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
# Learning tool implementations
# ---------------------------------------------------------------------------


def _get_or_create_learner(name: str, **kwargs) -> Any:
    """Lazy-create a Learner on first use."""
    if name not in _learners:
        from qortex.learning import Learner, LearnerConfig

        config = LearnerConfig(name=name, state_dir=_learning_state_dir, **kwargs)
        _learners[name] = Learner(config)
    return _learners[name]


def _learning_select_impl(
    learner: str,
    candidates: list[dict],
    context: dict | None = None,
    k: int = 1,
    token_budget: int = 0,
    min_pulls: int = 0,
    seed_arms: list[str] | None = None,
    seed_boost: float | None = None,
) -> dict:
    """Select arms from candidates using the learner's strategy."""
    _ensure_initialized()
    from qortex.learning import Arm

    arms = [
        Arm(
            id=c["id"],
            metadata=c.get("metadata", {}),
            token_cost=c.get("token_cost", 0),
        )
        for c in candidates
    ]

    create_kwargs: dict = {}
    if seed_arms is not None:
        create_kwargs["seed_arms"] = seed_arms
    if seed_boost is not None:
        create_kwargs["seed_boost"] = seed_boost
    lrn = _get_or_create_learner(learner, **create_kwargs)
    if min_pulls > 0:
        lrn.config.min_pulls = min_pulls
    result = lrn.select(arms, context=context, k=k, token_budget=token_budget)

    return {
        "selected_arms": [
            {"id": a.id, "metadata": a.metadata, "token_cost": a.token_cost}
            for a in result.selected
        ],
        "excluded_arms": [
            {"id": a.id, "metadata": a.metadata, "token_cost": a.token_cost}
            for a in result.excluded
        ],
        "is_baseline": result.is_baseline,
        "scores": {k: round(v, 4) for k, v in result.scores.items()},
        "token_budget": result.token_budget,
        "used_tokens": result.used_tokens,
    }


def _learning_observe_impl(
    learner: str,
    arm_id: str,
    outcome: str = "",
    reward: float = 0.0,
    context: dict | None = None,
) -> dict:
    """Record an observation and update posterior."""
    _ensure_initialized()
    from qortex.learning import ArmOutcome

    lrn = _get_or_create_learner(learner)
    obs = ArmOutcome(
        arm_id=arm_id,
        reward=reward,
        outcome=outcome,
        context=context or {},
    )
    state = lrn.observe(obs, context=context)

    return {
        "arm_id": arm_id,
        "alpha": round(state.alpha, 4),
        "beta": round(state.beta, 4),
        "mean": round(state.mean, 4),
        "pulls": state.pulls,
    }


def _learning_posteriors_impl(
    learner: str,
    context: dict | None = None,
    arm_ids: list[str] | None = None,
) -> dict:
    """Get current posteriors for arms."""
    _ensure_initialized()
    lrn = _get_or_create_learner(learner)
    posteriors = lrn.posteriors(context=context, arm_ids=arm_ids)

    return {
        "learner": learner,
        "posteriors": {
            arm_id: {k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()}
            for arm_id, p in posteriors.items()
        },
    }


def _learning_metrics_impl(
    learner: str,
    window: int | None = None,
) -> dict:
    """Get learning metrics."""
    _ensure_initialized()
    lrn = _get_or_create_learner(learner)
    return lrn.metrics(window=window)


def _learning_session_start_impl(
    learner: str,
    session_name: str,
) -> dict:
    """Start a named learning session."""
    _ensure_initialized()
    lrn = _get_or_create_learner(learner)
    session_id = lrn.session_start(session_name)
    return {"session_id": session_id, "learner": learner}


def _learning_session_end_impl(session_id: str) -> dict:
    """End a learning session and return summary."""
    _ensure_initialized()
    for lrn in _learners.values():
        if session_id in lrn._sessions:
            return lrn.session_end(session_id)
    return {"error": f"Session {session_id} not found in any learner"}


def _learning_reset_impl(
    learner: str,
    arm_ids: list[str] | None = None,
    context: dict | None = None,
) -> dict:
    """Reset (delete) learned posteriors for a learner."""
    _ensure_initialized()
    lrn = _get_or_create_learner(learner)
    count = lrn.reset(arm_ids=arm_ids, context=context)
    # Evict from cache so seed boosts re-apply on next use
    _learners.pop(learner, None)
    return {"learner": learner, "deleted": count, "status": "reset"}


# ---------------------------------------------------------------------------
# Learning MCP tool wrappers
# ---------------------------------------------------------------------------


@mcp.tool
@_mcp_traced
def qortex_learning_select(
    learner: str,
    candidates: list[dict],
    context: dict | None = None,
    k: int = 1,
    token_budget: int = 0,
    min_pulls: int = 0,
    seed_arms: list[str] | None = None,
    seed_boost: float | None = None,
) -> dict:
    """Select the best candidates from a pool using adaptive learning.

    Balances exploring new options vs exploiting known-good ones. Returns
    selections ranked by learned quality. Call qortex_learning_observe
    after using selections to close the feedback loop.

    Args:
        learner: Learner name (auto-created on first use).
        candidates: List of dicts, each with "id" (str). Optional: "metadata", "token_cost".
        context: Optional context dict for partitioned learning.
        k: Number of arms to select.
        token_budget: Max total token cost. 0 = unlimited.
        min_pulls: Force-explore arms with fewer than this many observations.
        seed_arms: Arm IDs to boost on first use. Applied when the learner is created.
        seed_boost: Alpha boost for seed arms (default 2.0 in LearnerConfig).
    """
    return _learning_select_impl(
        learner, candidates, context, k, token_budget, min_pulls,
        seed_arms=seed_arms, seed_boost=seed_boost,
    )


@mcp.tool
@_mcp_traced
def qortex_learning_observe(
    learner: str,
    arm_id: str,
    outcome: str = "",
    reward: float = 0.0,
    context: dict | None = None,
) -> dict:
    """Record whether a selected item worked well.

    Updates the system's beliefs about item quality. Always call this
    after qortex_learning_select to close the feedback loop.

    Args:
        learner: Learner name.
        arm_id: The arm ID that was used.
        outcome: "accepted", "rejected", or "partial".
        reward: Direct reward 0.0-1.0. Overrides outcome if both given.
        context: Context dict matching the select call.
    """
    return _learning_observe_impl(learner, arm_id, outcome, reward, context)


@mcp.tool
@_mcp_traced
def qortex_learning_posteriors(
    learner: str,
    context: dict | None = None,
    arm_ids: list[str] | None = None,
) -> dict:
    """Get current posterior distributions for arms.

    Returns alpha, beta, mean, and pull count for each tracked arm.

    Args:
        learner: Learner name.
        context: Optional context filter.
        arm_ids: Optional list of arm IDs to filter. None = all.
    """
    return _learning_posteriors_impl(learner, context, arm_ids)


@mcp.tool
@_mcp_traced
def qortex_learning_metrics(
    learner: str,
    window: int | None = None,
) -> dict:
    """Get aggregate learning metrics.

    Returns accuracy, total pulls, reward totals, and exploration ratio.

    Args:
        learner: Learner name.
        window: Optional window size (not yet implemented, reserved).
    """
    return _learning_metrics_impl(learner, window)


@mcp.tool
@_mcp_traced
def qortex_learning_session_start(
    learner: str,
    session_name: str,
) -> dict:
    """Start a named learning session for tracking arm selections.

    Args:
        learner: Learner name.
        session_name: Human-readable session name.
    """
    return _learning_session_start_impl(learner, session_name)


@mcp.tool
@_mcp_traced
def qortex_learning_session_end(session_id: str) -> dict:
    """End a learning session and return summary.

    Args:
        session_id: The session_id from qortex_learning_session_start.
    """
    return _learning_session_end_impl(session_id)


@mcp.tool
@_mcp_traced
def qortex_learning_reset(
    learner: str,
    arm_ids: list[str] | None = None,
    context: dict | None = None,
) -> dict:
    """Reset learned posteriors for a learner, clearing poisoned or stale data.

    Deletes stored arm states and evicts the learner from cache so seed
    boosts re-apply on next use. Scope the reset with arm_ids and/or context.

    Args:
        learner: Learner name.
        arm_ids: Optional list of arm IDs to delete. None = all arms.
        context: Optional context dict to scope deletion. None = default context (or all if arm_ids also None).
    """
    return _learning_reset_impl(learner, arm_ids, context)


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
        from qortex.ingest.backends.anthropic import AnthropicBackend

        _llm_backend = AnthropicBackend()
        return _llm_backend
    except ImportError:
        pass
    except Exception as e:
        logger.warning("anthropic.init.failed", error=str(e))

    # Fallback to stub
    from qortex.ingest.base import StubLLMBackend

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
