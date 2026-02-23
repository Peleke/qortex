"""QortexService: shared service layer for MCP server and REST API.

Consolidates the module-level globals and _ensure_initialized() logic from
mcp/server.py into a proper object. Both the MCP server and REST API share
a QortexService instance.

Methods return plain dicts (JSON-serializable). The REST API passes them
straight to JSONResponse. HttpQortexClient deserializes them into protocol
result types (QueryResult, etc.).
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import os
from collections import deque
from pathlib import Path
from typing import Any

from qortex.observe.logging import get_logger

logger = get_logger(__name__)


def create_vec_index(type_str: str, dimensions: int = 384) -> Any:
    """Construct a VectorIndex from type string + env vars.

    Shared factory used by QortexService and MCP server.
    """
    if type_str == "sqlite":
        from qortex.vec.index import SqliteVecIndex

        vec_path = Path("~/.qortex/vectors.db").expanduser()
        return SqliteVecIndex(db_path=str(vec_path), dimensions=dimensions)
    elif type_str == "pgvector":
        from qortex.vec.pgvector import PgVectorIndex

        dsn = os.environ.get("PGVECTOR_DSN")
        if dsn is None:
            host = os.environ.get("PGVECTOR_HOST", "localhost")
            port = os.environ.get("PGVECTOR_PORT", "5432")
            user = os.environ.get("PGVECTOR_USER", "qortex")
            password = os.environ.get("PGVECTOR_PASSWORD", "qortex")
            db = os.environ.get("PGVECTOR_DB", "qortex")
            dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"
        return PgVectorIndex(dsn=dsn, dimensions=dimensions)
    elif type_str in ("numpy", "memory"):
        from qortex.vec.index import NumpyVectorIndex

        return NumpyVectorIndex(dimensions=dimensions)
    else:
        raise ValueError(
            f"Unknown vec backend type: {type_str!r}. "
            "Must be 'sqlite', 'pgvector', 'numpy', or 'memory'."
        )


def _build_dsn() -> str:
    """Construct postgres DSN from PGVECTOR_* env vars."""
    dsn = os.environ.get("PGVECTOR_DSN")
    if dsn is not None:
        return dsn
    host = os.environ.get("PGVECTOR_HOST", "localhost")
    port = os.environ.get("PGVECTOR_PORT", "5432")
    user = os.environ.get("PGVECTOR_USER", "qortex")
    password = os.environ.get("PGVECTOR_PASSWORD", "qortex")
    db = os.environ.get("PGVECTOR_DB", "qortex")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


class QortexService:
    """Encapsulates qortex backend state and operations.

    Shared by REST API, MCP server, and (optionally) LocalQortexClient.
    Replaces the module-level globals in mcp/server.py.
    """

    def __init__(
        self,
        backend: Any = None,
        vector_index: Any = None,
        embedding_model: Any = None,
        interoception: Any = None,
        llm_backend: Any = None,
        learning_state_dir: str = "",
        pg_pool: Any = None,
    ) -> None:
        from qortex.sources.registry import SourceRegistry

        self.backend = backend
        self.vector_index = vector_index
        self.embedding_model = embedding_model
        self.interoception = interoception
        self.llm_backend = llm_backend
        self.learning_state_dir = learning_state_dir
        self.pg_pool = pg_pool

        # Adapters (composed from vec + graph layers)
        self.adapter: Any = None  # VecOnlyAdapter
        self.graph_adapter: Any = None  # GraphRAGAdapter

        # Source registry
        self.source_registry = SourceRegistry()

        # Learning layer
        self.learners: dict[str, Any] = {}

        # Activity counters (not thread-safe; OK for single-worker)
        self.query_count: int = 0
        self.feedback_count: int = 0
        self.feedback_outcomes: dict[str, int] = {
            "accepted": 0,
            "rejected": 0,
            "partial": 0,
        }

        # Vector-level index registry (for MastraVector / raw vector ops)
        self.vector_indexes: dict[str, Any] = {}
        self.index_configs: dict[str, dict] = {}
        self.vector_metadata: dict[str, dict[str, dict]] = {}
        self.vector_documents: dict[str, dict[str, str]] = {}

        # Compose adapters if we have the pieces
        self._compose_adapters()

    def _compose_adapters(self) -> None:
        """Build vec-only and graph-enhanced adapters from current layers."""
        if (
            self.vector_index is not None
            and self.backend is not None
            and self.embedding_model is not None
        ):
            from qortex.hippocampus.adapter import GraphRAGAdapter, VecOnlyAdapter

            self.adapter = VecOnlyAdapter(
                self.vector_index, self.backend, self.embedding_model
            )
            self.graph_adapter = GraphRAGAdapter(
                self.vector_index,
                self.backend,
                self.embedding_model,
                interoception=self.interoception,
            )
        else:
            self.adapter = None
            self.graph_adapter = None

    @classmethod
    def from_env(cls) -> QortexService:
        """Create from environment variables (QORTEX_VEC, QORTEX_GRAPH, etc.).

        Consolidates _ensure_initialized() logic from mcp/server.py.
        """
        from qortex.core.memory import InMemoryBackend
        from qortex.observe import configure

        configure()

        # --- Embedding model ---
        embedding_model = None
        try:
            from qortex.vec.embeddings import SentenceTransformerEmbedding

            embedding_model = SentenceTransformerEmbedding()
            _ = embedding_model.dimensions  # verify loadable
        except (ImportError, Exception) as e:
            logger.warning("vec.unavailable", error=str(e))
            embedding_model = None

        # --- Vec layer ---
        vector_index = None
        if embedding_model is not None:
            vec_backend = os.environ.get("QORTEX_VEC", "sqlite")
            if vec_backend == "pgvector":
                try:
                    from qortex.vec.pgvector import PgVectorIndex

                    dsn = os.environ.get("PGVECTOR_DSN")
                    if dsn is None:
                        host = os.environ.get("PGVECTOR_HOST", "localhost")
                        port = os.environ.get("PGVECTOR_PORT", "5432")
                        user = os.environ.get("PGVECTOR_USER", "qortex")
                        password = os.environ.get("PGVECTOR_PASSWORD", "qortex")
                        db = os.environ.get("PGVECTOR_DB", "qortex")
                        dsn = f"postgresql://{user}:{password}@{host}:{port}/{db}"
                    vector_index = PgVectorIndex(
                        dsn=dsn,
                        dimensions=embedding_model.dimensions,
                    )
                except ImportError:
                    logger.warning(
                        "pgvector.unavailable", fallback="SqliteVecIndex"
                    )
                    vec_backend = "sqlite"  # fall through to sqlite
            if vec_backend == "sqlite":
                try:
                    from qortex.vec.index import SqliteVecIndex

                    vec_path = Path("~/.qortex/vectors.db").expanduser()
                    vector_index = SqliteVecIndex(
                        db_path=str(vec_path),
                        dimensions=embedding_model.dimensions,
                    )
                except ImportError:
                    logger.warning(
                        "sqlite-vec.unavailable", fallback="NumpyVectorIndex"
                    )
                    from qortex.vec.index import NumpyVectorIndex

                    vector_index = NumpyVectorIndex(
                        dimensions=embedding_model.dimensions
                    )
            elif vec_backend == "memory":
                from qortex.vec.index import NumpyVectorIndex

                vector_index = NumpyVectorIndex(
                    dimensions=embedding_model.dimensions
                )

        # --- Graph layer ---
        backend = None
        graph_backend = os.environ.get("QORTEX_GRAPH", "memory")
        if graph_backend == "memgraph":
            try:
                from qortex.core.backend import MemgraphBackend, MemgraphCredentials

                host = os.environ.get("MEMGRAPH_HOST", "localhost")
                port = int(os.environ.get("MEMGRAPH_PORT", "7687"))
                user = os.environ.get("MEMGRAPH_USER", "")
                password = os.environ.get("MEMGRAPH_PASSWORD", "")
                creds = (
                    MemgraphCredentials(user=user, password=password) if user else None
                )
                backend = MemgraphBackend(host=host, port=port, credentials=creds)
                backend.connect()
            except (ImportError, Exception) as e:
                logger.warning(
                    "memgraph.unavailable",
                    error=str(e),
                    fallback="InMemoryBackend",
                )
                backend = InMemoryBackend(vector_index=vector_index)
                backend.connect()
        else:  # "memory"
            backend = InMemoryBackend(vector_index=vector_index)
            backend.connect()

        # --- Interoception ---
        from qortex.hippocampus.interoception import (
            InteroceptionConfig,
            LocalInteroceptionProvider,
        )

        qortex_dir = Path("~/.qortex").expanduser()
        interoception_config = InteroceptionConfig(
            db_path=qortex_dir / "interoception.db",
        )
        interoception = LocalInteroceptionProvider(interoception_config)
        interoception.startup()

        service = cls(
            backend=backend,
            vector_index=vector_index,
            embedding_model=embedding_model,
            interoception=interoception,
        )

        def _shutdown():
            try:
                interoception.shutdown()
            except Exception:
                logger.warning("interoception.shutdown.failed", exc_info=True)

        atexit.register(_shutdown)

        return service

    @classmethod
    async def async_from_env(cls) -> QortexService:
        """Create from env vars with async postgres stores.

        When QORTEX_STORE=postgres, uses shared asyncpg pool for:
        - PgVectorIndex (if QORTEX_VEC=pgvector)
        - AsyncLocalInteroceptionProvider
        - PostgresLearningStore (via pg_pool passed to learner creation)

        Falls back to from_env() when QORTEX_STORE != postgres.
        """
        store_backend = os.environ.get("QORTEX_STORE", "sqlite")
        if store_backend != "postgres":
            return cls.from_env()

        from qortex.core.memory import InMemoryBackend
        from qortex.core.pool import get_shared_pool
        from qortex.observe import configure

        configure()

        # --- Shared pool ---
        dsn = _build_dsn()

        async def _init_connection(conn):
            try:
                from pgvector.asyncpg import register_vector

                await register_vector(conn)
            except ImportError:
                pass

        pool = await get_shared_pool(dsn, init=_init_connection)

        # --- Embedding model ---
        embedding_model = None
        try:
            from qortex.vec.embeddings import SentenceTransformerEmbedding

            embedding_model = SentenceTransformerEmbedding()
            _ = embedding_model.dimensions
        except (ImportError, Exception) as e:
            logger.warning("vec.unavailable", error=str(e))
            embedding_model = None

        # --- Vec layer (shared pool) ---
        vector_index = None
        if embedding_model is not None:
            vec_backend = os.environ.get("QORTEX_VEC", "pgvector")
            if vec_backend == "pgvector":
                try:
                    from qortex.vec.pgvector import PgVectorIndex

                    vector_index = PgVectorIndex(
                        dsn=dsn,
                        dimensions=embedding_model.dimensions,
                        pool=pool,
                    )
                except ImportError:
                    logger.warning("pgvector.unavailable", fallback="NumpyVectorIndex")
                    from qortex.vec.index import NumpyVectorIndex

                    vector_index = NumpyVectorIndex(dimensions=embedding_model.dimensions)
            elif vec_backend == "sqlite":
                from qortex.vec.index import SqliteVecIndex

                vec_path = Path("~/.qortex/vectors.db").expanduser()
                vector_index = SqliteVecIndex(
                    db_path=str(vec_path), dimensions=embedding_model.dimensions
                )
            else:
                from qortex.vec.index import NumpyVectorIndex

                vector_index = NumpyVectorIndex(dimensions=embedding_model.dimensions)

        # --- Graph layer ---
        backend = None
        graph_backend = os.environ.get("QORTEX_GRAPH", "memory")
        if graph_backend == "memgraph":
            try:
                from qortex.core.backend import MemgraphBackend, MemgraphCredentials

                host = os.environ.get("MEMGRAPH_HOST", "localhost")
                port = int(os.environ.get("MEMGRAPH_PORT", "7687"))
                user = os.environ.get("MEMGRAPH_USER", "")
                password = os.environ.get("MEMGRAPH_PASSWORD", "")
                creds = MemgraphCredentials(user=user, password=password) if user else None
                backend = MemgraphBackend(host=host, port=port, credentials=creds)
                backend.connect()
            except (ImportError, Exception) as e:
                logger.warning(
                    "memgraph.unavailable", error=str(e), fallback="InMemoryBackend"
                )
                backend = InMemoryBackend(vector_index=vector_index)
                backend.connect()
        else:
            backend = InMemoryBackend(vector_index=vector_index)
            backend.connect()

        # --- Interoception (postgres) ---
        from qortex.hippocampus.interoception import (
            AsyncLocalInteroceptionProvider,
            InteroceptionConfig,
        )

        interoception_config = InteroceptionConfig(postgres_pool=pool)
        interoception = AsyncLocalInteroceptionProvider(interoception_config)
        await interoception.startup()

        service = cls(
            backend=backend,
            vector_index=vector_index,
            embedding_model=embedding_model,
            interoception=interoception,
            pg_pool=pool,
        )

        return service

    # ------------------------------------------------------------------
    # Adapter selection
    # ------------------------------------------------------------------

    def select_adapter(self, mode: str = "auto") -> Any:
        """Select the right adapter based on mode."""
        if mode == "vec":
            return self.adapter
        if mode == "graph":
            return self.graph_adapter or self.adapter
        return self.graph_adapter or self.adapter

    # ------------------------------------------------------------------
    # Core operations (return dicts)
    # ------------------------------------------------------------------

    async def query(
        self,
        context: str,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
        mode: str = "auto",
    ) -> dict:
        top_k = max(1, min(top_k, 1000))
        min_confidence = max(0.0, min(min_confidence, 1.0))

        adapter = self.select_adapter(mode)
        if adapter is None:
            return {
                "items": [],
                "query_id": "",
                "error": "No embedding model available. Install qortex[vec].",
            }

        result = await adapter.retrieve(
            query=context,
            domains=domains,
            top_k=top_k,
            min_confidence=min_confidence,
        )

        self.query_count += 1

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

        rules = self._collect_query_rules(items, domains)

        return {
            "items": items,
            "query_id": result.query_id,
            "rules": rules,
        }

    _ALLOWED_OUTCOMES = {"accepted", "rejected", "partial"}
    _OUTCOME_REWARD = {"accepted": 1.0, "rejected": -1.0, "partial": 0.3}

    async def feedback(
        self,
        query_id: str,
        outcomes: dict[str, str],
        source: str = "unknown",
    ) -> dict:
        for item_id, outcome in outcomes.items():
            if outcome not in self._ALLOWED_OUTCOMES:
                return {
                    "error": f"Invalid outcome '{outcome}' for item '{item_id}'. "
                    f"Must be one of: {', '.join(sorted(self._ALLOWED_OUTCOMES))}",
                }

        if self.graph_adapter is not None:
            self.graph_adapter.feedback(query_id, outcomes)
        elif self.adapter is not None:
            self.adapter.feedback(query_id, outcomes)

        self.feedback_count += 1
        for outcome in outcomes.values():
            if outcome in self.feedback_outcomes:
                self.feedback_outcomes[outcome] += 1

        credit_summary = await self._maybe_propagate_credit(query_id, outcomes)

        result: dict = {
            "status": "recorded",
            "query_id": query_id,
            "outcome_count": len(outcomes),
            "source": source,
        }
        if credit_summary is not None:
            result["credit"] = credit_summary

        return result

    async def ingest(
        self,
        source_path: str,
        domain: str,
        source_type: str | None = None,
    ) -> dict:
        path = Path(source_path).expanduser().resolve()
        if not path.exists():
            return {"error": f"File not found: {source_path}"}
        if not path.is_file():
            return {"error": f"Not a file: {source_path}"}

        allowed = {"text", "markdown", "pdf"}
        if source_type is not None and source_type not in allowed:
            return {
                "error": f"Invalid source_type: {source_type}. Must be one of {allowed}"
            }

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

        source = Source(path=path, source_type=source_type, name=path.name)
        llm = self._get_llm_backend()

        if source_type == "markdown":
            from qortex.ingest.markdown import MarkdownIngestor

            ingestor = MarkdownIngestor(llm, embedding_model=self.embedding_model)
        elif source_type == "pdf":
            from qortex.ingest.pdf import PDFIngestor

            ingestor = PDFIngestor(llm, embedding_model=self.embedding_model)
        else:
            from qortex.ingest.text import TextIngestor

            ingestor = TextIngestor(llm, embedding_model=self.embedding_model)

        manifest = ingestor.ingest(source, domain=domain)
        self.backend.ingest_manifest(manifest)
        await self._index_manifest_embeddings(manifest)

        return {
            "domain": domain,
            "source": path.name,
            "concepts": len(manifest.concepts),
            "edges": len(manifest.edges),
            "rules": len(manifest.rules),
            "warnings": manifest.warnings,
        }

    async def ingest_text(
        self,
        text: str,
        domain: str,
        format: str = "text",
        name: str | None = None,
    ) -> dict:
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

        source = Source(raw_content=text, source_type=format, name=name or "raw_text")
        llm = self._get_llm_backend()

        if format == "markdown":
            from qortex.ingest.markdown import MarkdownIngestor

            ingestor = MarkdownIngestor(llm, embedding_model=self.embedding_model)
        else:
            from qortex.ingest.text import TextIngestor

            ingestor = TextIngestor(llm, embedding_model=self.embedding_model)

        manifest = ingestor.ingest(source, domain=domain)
        self.backend.ingest_manifest(manifest)
        await self._index_manifest_embeddings(manifest)

        return {
            "domain": domain,
            "source": name or "raw_text",
            "concepts": len(manifest.concepts),
            "edges": len(manifest.edges),
            "rules": len(manifest.rules),
            "warnings": manifest.warnings,
        }

    async def ingest_structured(
        self,
        concepts: list[dict],
        domain: str,
        edges: list[dict] | None = None,
        rules: list[dict] | None = None,
    ) -> dict:
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

        if self.backend.get_domain(domain) is None:
            self.backend.create_domain(domain)

        source_id = f"structured:{hashlib.sha256(domain.encode()).hexdigest()[:12]}"

        concept_nodes: list[ConceptNode] = []
        name_to_id: dict[str, str] = {}
        for c in concepts:
            cname = c.get("name")
            if not cname:
                continue
            desc = c.get("description", cname)
            node_id = c.get(
                "id",
                f"{domain}:{hashlib.sha256(cname.encode()).hexdigest()[:12]}",
            )
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

        if self.embedding_model is not None and concept_nodes:
            texts = [f"{n.name}: {n.description}" for n in concept_nodes]
            embs = self.embedding_model.embed(texts)
            for node, emb in zip(concept_nodes, embs):
                node.embedding = emb

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

        explicit_rules: list[ExplicitRule] = []
        for r in rules:
            rtext = r.get("text", "")
            if not rtext:
                continue
            rule_id = (
                f"{domain}:rule:{hashlib.sha256(rtext.encode()).hexdigest()[:12]}"
            )
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

        self.backend.ingest_manifest(manifest)
        await self._index_manifest_embeddings(manifest)

        return {
            "domain": domain,
            "source": "structured_input",
            "concepts": len(concept_nodes),
            "edges": len(concept_edges),
            "rules": len(explicit_rules),
        }

    async def ingest_message(
        self,
        text: str,
        session_id: str,
        role: str = "user",
        domain: str = "session",
    ) -> dict:
        """Lightweight online-index pipeline: chunk, embed, index.

        No LLM required. Used by the gateway to index session messages
        for retrieval during conversations.
        """
        import time

        _VALID_ROLES = {"user", "assistant", "system", "tool"}

        if not text or not text.strip():
            return {"session_id": session_id, "chunks": 0, "concepts": 0, "edges": 0}

        safe_role = role if role in _VALID_ROLES else "unknown"
        t0 = time.monotonic()

        # Chunk
        from qortex.online.chunker import default_chunker

        chunks = default_chunker(text, source_id=f"{session_id}:{safe_role}")

        chunk_count = 0
        if self.embedding_model is not None and self.vector_index is not None and chunks:
            ids = [f"{session_id}:{c.id}" for c in chunks]
            texts = [c.text for c in chunks]
            embeddings = self.embedding_model.embed(texts)
            await self.vector_index.add(ids, embeddings)
            chunk_count = len(chunks)

        elapsed = (time.monotonic() - t0) * 1000

        try:
            from qortex.observe.emitter import emit
            from qortex.observe.events import MessageIngested

            emit(
                MessageIngested(
                    session_id=session_id,
                    role=safe_role,
                    domain=domain,
                    chunk_count=chunk_count,
                    concept_count=0,
                    edge_count=0,
                    latency_ms=elapsed,
                )
            )
        except Exception:
            pass

        return {
            "session_id": session_id,
            "chunks": chunk_count,
            "concepts": 0,
            "edges": 0,
            "latency_ms": round(elapsed, 2),
        }

    async def domains(self) -> dict:
        ds = self.backend.list_domains()
        return {
            "domains": [
                {
                    "name": d.name,
                    "description": d.description,
                    "concept_count": d.concept_count,
                    "edge_count": d.edge_count,
                    "rule_count": d.rule_count,
                }
                for d in ds
            ]
        }

    async def status(self) -> dict:
        backend_type = type(self.backend).__name__
        has_vec = self.vector_index is not None
        has_mage = self.backend.supports_mage() if self.backend else False
        domain_count = len(self.backend.list_domains()) if self.backend else 0

        return {
            "status": "ok",
            "backend": backend_type,
            "vector_index": (
                type(self.vector_index).__name__ if self.vector_index else None
            ),
            "vector_search": has_vec,
            "graph_algorithms": has_mage,
            "domain_count": domain_count,
            "embedding_model": (
                getattr(
                    self.embedding_model,
                    "_model_name",
                    type(self.embedding_model).__name__,
                )
                if self.embedding_model
                else None
            ),
            "interoception": (
                self.interoception.summary() if self.interoception else None
            ),
        }

    async def explore(self, node_id: str, depth: int = 1) -> dict | None:
        depth = max(1, min(depth, 3))

        node = self.backend.get_node(node_id)
        if node is None:
            return None

        visited: set[str] = {node_id}
        all_edges: list[dict] = []
        all_neighbors: list[dict] = []
        frontier: deque[str] = deque([node_id])

        for _hop in range(depth):
            next_frontier: deque[str] = deque()
            while frontier:
                current_id = frontier.popleft()
                edges = list(self.backend.get_edges(current_id, "both"))
                for edge in edges:
                    edge_dict = {
                        "source_id": edge.source_id,
                        "target_id": edge.target_id,
                        "relation_type": (
                            edge.relation_type.value
                            if hasattr(edge.relation_type, "value")
                            else str(edge.relation_type)
                        ),
                        "confidence": edge.confidence,
                        "properties": edge.properties,
                    }
                    all_edges.append(edge_dict)
                    neighbor_id = (
                        edge.target_id
                        if edge.source_id == current_id
                        else edge.source_id
                    )
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        neighbor_node = self.backend.get_node(neighbor_id)
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

        from qortex.core.rules import collect_rules_for_concepts

        rules = collect_rules_for_concepts(self.backend, list(visited))
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

    async def rules(
        self,
        domains: list[str] | None = None,
        concept_ids: list[str] | None = None,
        categories: list[str] | None = None,
        include_derived: bool = True,
        min_confidence: float = 0.0,
    ) -> dict:
        from qortex.projectors.models import ProjectionFilter
        from qortex.projectors.sources.flat import FlatRuleSource

        filt = ProjectionFilter(
            domains=domains,
            categories=categories,
            min_confidence=min_confidence,
        )

        source = FlatRuleSource(
            backend=self.backend, include_derived=include_derived
        )
        all_rules = source.derive(domains=domains, filters=filt)

        if concept_ids is not None:
            concept_set = set(concept_ids)
            all_rules = [
                r for r in all_rules if concept_set.intersection(r.source_concepts)
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

    async def compare(
        self,
        context: str,
        domains: list[str] | None = None,
        top_k: int = 5,
    ) -> dict:
        top_k = max(1, min(top_k, 20))

        if self.adapter is None:
            return {"error": "No vector index available. Install qortex[vec]."}

        vec_result = await self.adapter.retrieve(
            query=context, domains=domains, top_k=top_k, min_confidence=0.0
        )

        graph_result = None
        if self.graph_adapter is not None:
            graph_result = await self.graph_adapter.retrieve(
                query=context, domains=domains, top_k=top_k, min_confidence=0.0
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

        rules = (
            self._collect_query_rules(graph_items, domains) if graph_result else []
        )

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
            "summary": self._compare_summary(
                vec_items, graph_items, graph_unique, rules
            ),
        }

    async def stats(self) -> dict:
        ds = self.backend.list_domains() if self.backend else []
        domain_breakdown = {}
        totals = {"concepts": 0, "edges": 0, "rules": 0}
        for d in ds:
            s = {
                "concepts": d.concept_count,
                "edges": d.edge_count,
                "rules": d.rule_count,
            }
            domain_breakdown[d.name] = s
            for k in totals:
                totals[k] += s[k]

        learner_breakdown = {}
        total_observations = 0
        for name, lrn in self.learners.items():
            m = await lrn.metrics()
            learner_breakdown[name] = {
                "total_pulls": m.get("total_pulls", 0),
                "accuracy": round(m.get("accuracy", 0.0), 3),
                "arms_tracked": m.get("arm_count", 0),
                "exploration_ratio": round(m.get("explore_ratio", 0.0), 3),
            }
            total_observations += m.get("total_pulls", 0)

        feedback_rate = (
            round(self.feedback_count / self.query_count, 3)
            if self.query_count > 0
            else 0.0
        )

        return {
            "knowledge": {
                "domains": len(ds),
                **totals,
                "domain_breakdown": domain_breakdown,
            },
            "learning": {
                "learners": len(self.learners),
                "total_observations": total_observations,
                "learner_breakdown": learner_breakdown,
            },
            "activity": {
                "queries_served": self.query_count,
                "feedback_given": self.feedback_count,
                "feedback_rate": feedback_rate,
                "outcomes": dict(self.feedback_outcomes),
            },
            "health": {
                "backend": (
                    type(self.backend).__name__ if self.backend else None
                ),
                "vector_index": (
                    type(self.vector_index).__name__
                    if self.vector_index
                    else None
                ),
                "embedding_model": (
                    getattr(
                        self.embedding_model,
                        "_model_name",
                        type(self.embedding_model).__name__,
                    )
                    if self.embedding_model
                    else None
                ),
                "persistence": self._get_persistence_info(),
            },
        }

    # ------------------------------------------------------------------
    # Source operations
    # ------------------------------------------------------------------

    async def source_connect(
        self,
        source_id: str,
        connection_string: str,
        schemas: list[str] | None = None,
        domain_map: dict[str, str] | None = None,
    ) -> dict:
        from qortex.sources.base import SourceConfig

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
            await adapter.connect(config)
            table_schemas = await adapter.discover()
        except Exception as e:
            return {"error": f"Connection failed: {e}"}

        self.source_registry.register(config, adapter)
        self.source_registry.cache_schemas(source_id, table_schemas)

        return {
            "status": "connected",
            "source_id": source_id,
            "tables": len(table_schemas),
            "table_names": [t.name for t in table_schemas],
        }

    def source_discover(self, source_id: str) -> dict:
        schemas = self.source_registry.get_schemas(source_id)
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

    async def source_sync(
        self,
        source_id: str,
        tables: list[str] | None = None,
        mode: str = "full",
    ) -> dict:
        adapter = self.source_registry.get(source_id)
        config = self.source_registry.get_config(source_id)
        if adapter is None or config is None:
            return {"error": f"Source '{source_id}' not found. Connect first."}

        try:
            result = await adapter.sync(
                tables=tables,
                mode=mode,
                vector_index=self.vector_index,
                embedding_model=self.embedding_model,
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

    async def source_list(self) -> dict:
        sources = self.source_registry.list_sources()
        return {
            "sources": [
                {
                    "source_id": sid,
                    "tables": len(self.source_registry.get_schemas(sid) or []),
                }
                for sid in sources
            ]
        }

    async def source_disconnect(self, source_id: str) -> dict:
        adapter = self.source_registry.get(source_id)
        if adapter is None:
            return {"error": f"Source '{source_id}' not found."}

        await self.source_registry.remove_async(source_id)
        return {"status": "disconnected", "source_id": source_id}

    # ------------------------------------------------------------------
    # Learning operations
    # ------------------------------------------------------------------

    async def _get_or_create_learner(self, name: str, **kwargs) -> Any:
        if name not in self.learners:
            from qortex.learning import Learner, LearnerConfig

            config = LearnerConfig(
                name=name, state_dir=self.learning_state_dir, **kwargs
            )
            store = None
            if self.pg_pool is not None:
                from qortex.learning.pg_store import PostgresLearningStore

                store = PostgresLearningStore(name, self.pg_pool)
            self.learners[name] = await Learner.create(config, store=store)
        return self.learners[name]

    async def learning_select(
        self,
        learner: str,
        candidates: list[dict],
        context: dict | None = None,
        k: int = 1,
        token_budget: int = 0,
        min_pulls: int = 0,
        seed_arms: list[str] | None = None,
        seed_boost: float | None = None,
    ) -> dict:
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
        lrn = await self._get_or_create_learner(learner, **create_kwargs)
        if min_pulls > 0:
            lrn.config.min_pulls = min_pulls
        result = await lrn.select(
            arms, context=context, k=k, token_budget=token_budget
        )

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
            "scores": {k_: round(v, 4) for k_, v in result.scores.items()},
            "token_budget": result.token_budget,
            "used_tokens": result.used_tokens,
        }

    async def learning_observe(
        self,
        learner: str,
        arm_id: str,
        outcome: str = "",
        reward: float = 0.0,
        context: dict | None = None,
    ) -> dict:
        from qortex.learning import ArmOutcome

        lrn = await self._get_or_create_learner(learner)
        obs = ArmOutcome(
            arm_id=arm_id,
            reward=reward,
            outcome=outcome,
            context=context or {},
        )
        state = await lrn.observe(obs, context=context)

        return {
            "arm_id": arm_id,
            "alpha": round(state.alpha, 4),
            "beta": round(state.beta, 4),
            "mean": round(state.mean, 4),
            "pulls": state.pulls,
        }

    async def learning_posteriors(
        self,
        learner: str,
        context: dict | None = None,
        arm_ids: list[str] | None = None,
    ) -> dict:
        lrn = await self._get_or_create_learner(learner)
        posteriors = await lrn.posteriors(context=context, arm_ids=arm_ids)

        return {
            "learner": learner,
            "posteriors": {
                arm_id: {
                    k_: round(v, 4) if isinstance(v, float) else v
                    for k_, v in p.items()
                }
                for arm_id, p in posteriors.items()
            },
        }

    async def learning_metrics(
        self,
        learner: str,
        window: int | None = None,
    ) -> dict:
        lrn = await self._get_or_create_learner(learner)
        return await lrn.metrics(window=window)

    async def learning_session_start(
        self,
        learner: str,
        session_name: str,
    ) -> dict:
        lrn = await self._get_or_create_learner(learner)
        session_id = lrn.session_start(session_name)
        return {"session_id": session_id, "learner": learner}

    async def learning_session_end(self, session_id: str) -> dict:
        for lrn in self.learners.values():
            if session_id in lrn._sessions:
                return lrn.session_end(session_id)
        return {"error": f"Session {session_id} not found in any learner"}

    async def learning_reset(
        self,
        learner: str,
        arm_ids: list[str] | None = None,
        context: dict | None = None,
    ) -> dict:
        lrn = await self._get_or_create_learner(learner)
        count = await lrn.reset(arm_ids=arm_ids, context=context)
        self.learners.pop(learner, None)
        return {"learner": learner, "deleted": count, "status": "reset"}

    # ------------------------------------------------------------------
    # Vector-level operations (MastraVector / raw vector)
    # ------------------------------------------------------------------

    async def vector_create_index(
        self,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
    ) -> dict:
        if index_name in self.vector_indexes:
            existing = self.index_configs[index_name]
            if existing["dimension"] != dimension:
                return {
                    "error": f"Index '{index_name}' already exists with dimension "
                    f"{existing['dimension']}, requested {dimension}",
                }
            return {"status": "exists", "index_name": index_name}

        from qortex.vec.index import NumpyVectorIndex

        self.vector_indexes[index_name] = NumpyVectorIndex(dimensions=dimension)
        self.index_configs[index_name] = {"dimension": dimension, "metric": metric}
        self.vector_metadata[index_name] = {}
        self.vector_documents[index_name] = {}

        return {"status": "created", "index_name": index_name}

    async def vector_list_indexes(self) -> dict:
        return {"indexes": list(self.vector_indexes.keys())}

    async def vector_describe_index(self, index_name: str) -> dict:
        if index_name not in self.vector_indexes:
            return {"error": f"Index '{index_name}' not found"}
        idx = self.vector_indexes[index_name]
        cfg = self.index_configs[index_name]
        return {
            "dimension": cfg["dimension"],
            "count": await idx.size(),
            "metric": cfg.get("metric", "cosine"),
        }

    async def vector_delete_index(self, index_name: str) -> dict:
        if index_name not in self.vector_indexes:
            return {"error": f"Index '{index_name}' not found"}
        del self.vector_indexes[index_name]
        del self.index_configs[index_name]
        self.vector_metadata.pop(index_name, None)
        self.vector_documents.pop(index_name, None)
        return {"status": "deleted", "index_name": index_name}

    async def vector_upsert(
        self,
        index_name: str,
        vectors: list[list[float]],
        metadata: list[dict] | None = None,
        ids: list[str] | None = None,
        documents: list[str] | None = None,
    ) -> dict:
        import uuid

        if index_name not in self.vector_indexes:
            return {"error": f"Index '{index_name}' not found"}

        idx = self.vector_indexes[index_name]
        cfg = self.index_configs[index_name]
        generated_ids = ids or [str(uuid.uuid4()) for _ in vectors]

        if len(generated_ids) != len(vectors):
            return {
                "error": f"ids ({len(generated_ids)}) and vectors ({len(vectors)}) must match"
            }

        expected_dim = cfg["dimension"]
        for i, vec in enumerate(vectors):
            if len(vec) != expected_dim:
                return {
                    "error": f"Vector {i} has dimension {len(vec)}, expected {expected_dim}",
                }

        await idx.add(generated_ids, vectors)

        meta_store = self.vector_metadata[index_name]
        doc_store = self.vector_documents[index_name]
        meta_list = metadata or [{}] * len(vectors)
        doc_list: list[str | None] = (
            list(documents) if documents is not None else [None] * len(vectors)
        )

        for vid, meta, doc in zip(generated_ids, meta_list, doc_list):
            meta_store[vid] = meta or {}
            if doc is not None:
                doc_store[vid] = doc

        return {"ids": generated_ids}

    async def vector_query(
        self,
        index_name: str,
        query_vector: list[float],
        top_k: int = 10,
        filter: dict | None = None,
        include_vector: bool = False,
    ) -> dict:
        if index_name not in self.vector_indexes:
            return {"error": f"Index '{index_name}' not found"}

        idx = self.vector_indexes[index_name]
        cfg = self.index_configs[index_name]
        meta_store = self.vector_metadata[index_name]
        doc_store = self.vector_documents[index_name]

        expected_dim = cfg["dimension"]
        if len(query_vector) != expected_dim:
            return {
                "error": f"Query vector has dimension {len(query_vector)}, expected {expected_dim}",
            }

        fetch_k = top_k * 5 if filter else top_k
        raw_results = await idx.search(query_vector, top_k=fetch_k)

        results: list[dict] = []
        for vid, score in raw_results:
            meta = meta_store.get(vid, {})
            if filter and not self._match_filter(meta, filter):
                continue
            entry: dict = {
                "id": vid,
                "score": round(score, 4),
                "metadata": meta,
            }
            if vid in doc_store:
                entry["document"] = doc_store[vid]
            if include_vector:
                entry["vector"] = idx.get(vid)
            results.append(entry)
            if len(results) >= top_k:
                break

        return {"results": results}

    # ------------------------------------------------------------------
    # Migration operations
    # ------------------------------------------------------------------

    async def migrate_vec(
        self,
        source_type: str,
        *,
        batch_size: int = 500,
        dry_run: bool = False,
    ) -> dict:
        """Migrate vectors from another backend into the current one.

        Source credentials come from env vars, not request bodies.
        """
        from dataclasses import asdict

        from qortex.vec.migrate import migrate_vec

        source = self._create_vec_index(source_type)
        try:
            result = await migrate_vec(
                source,
                self.vector_index,
                batch_size=batch_size,
                dry_run=dry_run,
            )
            return asdict(result)
        finally:
            if hasattr(source, "close"):
                close = source.close
                if asyncio.iscoroutinefunction(close):
                    await close()
                else:
                    close()

    def _create_vec_index(self, type_str: str) -> Any:
        """Construct a VectorIndex from type string + env vars."""
        dims = 384
        if self.embedding_model is not None:
            dims = self.embedding_model.dimensions
        return create_vec_index(type_str, dims)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_query_rules(
        self,
        items: list[dict],
        domains: list[str] | None,
    ) -> list[dict]:
        if not items or self.backend is None:
            return []

        from qortex.core.rules import collect_rules_for_concepts

        activated_ids = [
            item["node_id"] for item in items if item.get("node_id")
        ]
        scores_map = {
            item["node_id"]: item["score"]
            for item in items
            if item.get("node_id")
        }

        rules = collect_rules_for_concepts(
            self.backend, activated_ids, domains, scores_map
        )
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

    async def _index_manifest_embeddings(self, manifest: Any) -> None:
        ids_with_embeddings = []
        embeddings_list = []
        for concept in manifest.concepts:
            if concept.embedding is not None:
                self.backend.add_embedding(concept.id, concept.embedding)
                ids_with_embeddings.append(concept.id)
                embeddings_list.append(concept.embedding)

        if self.vector_index is not None and ids_with_embeddings:
            await self.vector_index.add(ids_with_embeddings, embeddings_list)

    def _get_llm_backend(self) -> Any:
        if self.llm_backend is not None:
            return self.llm_backend

        try:
            from qortex.ingest.backends.anthropic import AnthropicBackend

            self.llm_backend = AnthropicBackend()
            return self.llm_backend
        except ImportError:
            pass
        except Exception as e:
            logger.warning("anthropic.init.failed", error=str(e))

        from qortex.ingest.base import StubLLMBackend

        self.llm_backend = StubLLMBackend()
        return self.llm_backend

    async def _maybe_propagate_credit(
        self,
        query_id: str,
        outcomes: dict[str, str],
    ) -> dict | None:
        from qortex.flags import get_flags

        if not get_flags().credit_propagation:
            return None

        try:
            from qortex.causal.credit import CreditAssigner
            from qortex.causal.dag import CausalDAG
        except ImportError:
            return None

        if not outcomes or self.backend is None:
            return None

        credit_domains: dict[str, list[str]] = {}
        for item_id in outcomes:
            node = self.backend.get_node(item_id)
            if node is not None:
                credit_domains.setdefault(node.domain, []).append(item_id)

        if not credit_domains:
            return None

        all_assignments = []
        for domain, concept_ids in credit_domains.items():
            try:
                dag = CausalDAG.from_backend(self.backend, domain)
            except Exception:
                continue

            rewards = [
                self._OUTCOME_REWARD.get(outcomes[cid], 0.0) for cid in concept_ids
            ]
            avg_reward = sum(rewards) / len(rewards)

            assigner = CreditAssigner(dag)
            assignments = assigner.assign_credit(concept_ids, avg_reward)
            all_assignments.extend(assignments)

        if not all_assignments:
            return None

        updates = CreditAssigner.to_posterior_updates(all_assignments)
        learner = await self._get_or_create_learner("credit")
        await learner.apply_credit_deltas(updates)

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

    @staticmethod
    def _compare_summary(
        vec_items: list, graph_items: list, graph_unique: list, rules: list
    ) -> str:
        if not graph_items:
            return "Graph adapter not available. Showing vector-only results."
        parts = []
        if graph_unique:
            parts.append(f"found {len(graph_unique)} item(s) that cosine missed")
        if rules:
            parts.append(f"surfaced {len(rules)} rule(s)")
        overlap = len(
            set(i["id"] for i in vec_items) & set(i["id"] for i in graph_items)
        )
        if overlap < len(vec_items):
            parts.append(f"replaced {len(vec_items) - overlap} distractor(s)")
        if not parts:
            return "Both methods returned similar results for this query."
        return "Graph-enhanced retrieval " + ", ".join(parts) + "."

    @staticmethod
    def _get_persistence_info() -> dict:
        state_dir = os.environ.get("QORTEX_STATE_DIR")
        if not state_dir:
            return {"mode": "in-memory", "persistent": False}
        state_path = Path(state_dir)
        if not state_path.exists():
            return {
                "mode": "configured",
                "path": str(state_path),
                "persistent": False,
            }
        files = list(state_path.glob("*.db"))
        size_bytes = sum(f.stat().st_size for f in files if f.is_file())
        return {
            "mode": "sqlite",
            "path": str(state_path),
            "persistent": True,
            "db_files": len(files),
            "size_mb": round(size_bytes / (1024 * 1024), 2),
        }

    @staticmethod
    def _match_filter(metadata: dict, filt: dict) -> bool:
        for key, value in filt.items():
            if key == "$and":
                if not all(
                    QortexService._match_filter(metadata, sub) for sub in value
                ):
                    return False
            elif key == "$or":
                if not any(
                    QortexService._match_filter(metadata, sub) for sub in value
                ):
                    return False
            elif key == "$not":
                if QortexService._match_filter(metadata, value):
                    return False
            elif isinstance(value, dict):
                actual = metadata.get(key)
                for op, operand in value.items():
                    if op == "$ne" and actual == operand:
                        return False
                    if op == "$gt" and not (actual is not None and actual > operand):
                        return False
                    if op == "$gte" and not (
                        actual is not None and actual >= operand
                    ):
                        return False
                    if op == "$lt" and not (actual is not None and actual < operand):
                        return False
                    if op == "$lte" and not (
                        actual is not None and actual <= operand
                    ):
                        return False
                    if op == "$in" and actual not in operand:
                        return False
                    if op == "$nin" and actual in operand:
                        return False
            else:
                if metadata.get(key) != value:
                    return False
        return True
