"""MCP server implementation for qortex.

Tools exposed:
- qortex_query: Retrieve relevant knowledge items for a context
- qortex_feedback: Report outcomes for retrieved items
- qortex_ingest: Ingest a file into a domain
- qortex_domains: List available domains
- qortex_status: Server health + backend info

Architecture:
    Each tool has a plain `_<name>_impl()` function with the core logic,
    plus an `@mcp.tool`-decorated wrapper that delegates to it.
    Tests call the `_impl` functions directly; MCP clients hit the wrappers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

from qortex.core.memory import InMemoryBackend
from qortex.hippocampus.adapter import VecOnlyAdapter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Server-level state: initialized once via create_server() or serve()
# ---------------------------------------------------------------------------

_backend: Any = None
_adapter: Any = None
_embedding_model: Any = None

mcp = FastMCP("qortex")


def _ensure_initialized() -> None:
    """Lazy initialization: set up backend + adapter if not already done."""
    global _backend, _adapter, _embedding_model

    if _backend is not None:
        return

    # Default: InMemoryBackend (no Docker required).
    # Production users configure Memgraph via env or create_server().
    _backend = InMemoryBackend()
    _backend.connect()

    # Try to load sentence-transformers for embeddings
    try:
        from qortex.vec.embeddings import SentenceTransformerEmbedding

        _embedding_model = SentenceTransformerEmbedding()
        _adapter = VecOnlyAdapter(_backend, _embedding_model)
    except ImportError:
        logger.warning("qortex[vec] not installed. Vector search unavailable.")
        _embedding_model = None
        _adapter = None


def create_server(
    backend: Any = None,
    embedding_model: Any = None,
) -> FastMCP:
    """Create and configure the MCP server with explicit backend/embedding model.

    Used by tests and advanced configurations. For normal usage, call serve().
    """
    global _backend, _adapter, _embedding_model

    _backend = backend
    _embedding_model = embedding_model

    if _backend is not None and _embedding_model is not None:
        _adapter = VecOnlyAdapter(_backend, _embedding_model)
    else:
        _adapter = None

    return mcp


# ---------------------------------------------------------------------------
# Tool implementations (plain functions — testable, no decorator wrapping)
# ---------------------------------------------------------------------------


def _query_impl(
    context: str,
    domains: list[str] | None = None,
    top_k: int = 20,
    min_confidence: float = 0.0,
) -> dict:
    _ensure_initialized()

    # Clamp inputs to valid ranges
    top_k = max(1, min(top_k, 1000))
    min_confidence = max(0.0, min(min_confidence, 1.0))

    if _adapter is None:
        return {
            "items": [],
            "query_id": "",
            "error": "No embedding model available. Install qortex[vec].",
        }

    result = _adapter.retrieve(
        query=context,
        domains=domains,
        top_k=top_k,
        min_confidence=min_confidence,
    )

    return {
        "items": [
            {
                "id": item.id,
                "content": item.content,
                "score": round(item.score, 4),
                "domain": item.domain,
                "node_id": item.node_id,
                "metadata": item.metadata,
            }
            for item in result.items
        ],
        "query_id": result.query_id,
    }


def _feedback_impl(
    query_id: str,
    outcomes: dict[str, str],
    source: str = "unknown",
) -> dict:
    _ensure_initialized()

    if _adapter is not None:
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

    # Index embeddings in backend
    for concept in manifest.concepts:
        if concept.embedding is not None:
            _backend.add_embedding(concept.id, concept.embedding)

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
    has_vec = _embedding_model is not None
    has_mage = _backend.supports_mage() if _backend else False
    domain_count = len(_backend.list_domains()) if _backend else 0

    return {
        "status": "ok",
        "backend": backend_type,
        "vector_search": has_vec,
        "graph_algorithms": has_mage,
        "domain_count": domain_count,
        "embedding_model": (
            getattr(_embedding_model, "_model_name", type(_embedding_model).__name__)
            if _embedding_model
            else None
        ),
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
) -> dict:
    """Retrieve relevant knowledge items for a context.

    Uses vector similarity search over concept embeddings.
    Returns items sorted by relevance score.

    Args:
        context: Query text (file content, task description, question).
        domains: Restrict search to these domains. None = search all.
        top_k: Maximum number of items to return.
        min_confidence: Minimum cosine similarity score (0.0 to 1.0).
    """
    return _query_impl(context, domains, top_k, min_confidence)


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
