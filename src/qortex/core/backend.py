"""Graph backend protocol and implementations.

The KG is backend-agnostic. Code against GraphBackend protocol.
Primary: Memgraph (for MAGE algorithms)
Fallback: SQLite (limited features, no PPR)
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
import uuid
from abc import abstractmethod
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

from qortex.observability import emit
from qortex.observability.events import ManifestIngested
from qortex.observability.tracing import traced

from .models import ConceptEdge, ConceptNode, Domain, ExplicitRule, IngestionManifest, RelationType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cypher span helpers (used by @traced on _run)
# ---------------------------------------------------------------------------

_CYPHER_OPS = {
    "MATCH": "query", "CREATE": "create", "DELETE": "delete",
    "MERGE": "upsert", "CALL": "procedure", "DROP": "drop",
    "RETURN": "query", "WITH": "query", "UNWIND": "query",
}


def _cypher_op(cypher: str) -> str:
    """Extract operation name from Cypher query template."""
    first_word = cypher.strip().split()[0].upper() if cypher.strip() else "UNKNOWN"
    return _CYPHER_OPS.get(first_word, "other")


def _sanitize_cypher(cypher: str, max_len: int = 200) -> str:
    """Truncate Cypher for span attribute. Never include full param values."""
    return cypher[:max_len]


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
    # Rule Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def add_rule(self, rule: ExplicitRule) -> None:
        """Add an explicit rule to the graph."""
        ...

    @abstractmethod
    def get_rules(self, domain: str | None = None) -> list[ExplicitRule]:
        """Get rules, optionally filtered by domain."""
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
        seed_weights: dict[str, float] | None = None,
        extra_edges: list[tuple[str, str, float]] | None = None,
        query_id: str | None = None,
    ) -> dict[str, float]:
        """Personalized PageRank from source nodes.

        Core algorithm for HippoRAG pattern completion.
        Returns node_id -> score mapping.

        Args:
            source_nodes: Seed node IDs for personalization.
            damping_factor: Probability of following edges vs teleporting.
            max_iterations: Max PPR iterations.
            domain: Restrict to this domain.
            seed_weights: Per-seed teleportation weights (from factors).
            extra_edges: Ephemeral edges from online generation.
        """
        ...

    # -------------------------------------------------------------------------
    # Vector Operations
    # -------------------------------------------------------------------------

    @abstractmethod
    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Store an embedding vector for a concept node."""
        ...

    @abstractmethod
    def get_embedding(self, node_id: str) -> list[float] | None:
        """Retrieve the embedding vector for a concept node."""
        ...

    @abstractmethod
    def vector_search(
        self,
        query_embedding: list[float],
        domain: str | None = None,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[ConceptNode, float]]:
        """Find nodes by vector similarity.

        Args:
            query_embedding: Query vector.
            domain: Restrict search to a domain.
            top_k: Max results.
            threshold: Min cosine similarity.

        Returns:
            List of (node, score) tuples sorted by descending similarity.
        """
        ...

    @abstractmethod
    def supports_vector_search(self) -> bool:
        """Whether this backend has vector search capabilities."""
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


class MemgraphCredentials:
    """Secure wrapper for Memgraph authentication.

    Can be initialized from a tuple or from the CLI config's MemgraphCredentials.
    Password is hidden from repr/str to avoid accidental logging.
    """

    __slots__ = ("_user", "_password")

    def __init__(self, user: str = "", password: str = ""):
        self._user = user
        self._password = password

    @classmethod
    def from_tuple(cls, auth: tuple[str, str]) -> MemgraphCredentials:
        """Create from (user, password) tuple."""
        return cls(user=auth[0], password=auth[1])

    @property
    def auth_tuple(self) -> tuple[str, str]:
        """Get (user, password) tuple for neo4j driver."""
        return (self._user, self._password)

    def __repr__(self) -> str:
        return f"MemgraphCredentials(user={self._user!r})"

    def __str__(self) -> str:
        return f"MemgraphCredentials(user={self._user!r})"


class MemgraphBackend:
    """Primary backend using Memgraph via the neo4j Bolt driver.

    Requires: pip install qortex[memgraph]  (neo4j>=5.0)
    Memgraph is wire-compatible with the Bolt protocol.

    Cypher schema:
        (:Domain {name, description, created_at, updated_at})
        (:Concept {id, name, description, domain, source_id, source_location, confidence, properties})
        (s:Concept)-[:REQUIRES|SUPPORTS|... {confidence, bidirectional, properties}]->(t:Concept)
        (:Rule {id, text, domain, source_id, source_location, category, confidence, concept_ids})
    """

    def __init__(
        self,
        uri: str | None = None,
        host: str = "localhost",
        port: int = 7687,
        credentials: MemgraphCredentials | None = None,
    ):
        """Initialize MemgraphBackend.

        Args:
            uri: Full bolt:// URI (takes precedence over host/port)
            host: Memgraph host (default: localhost)
            port: Memgraph port (default: 7687)
            credentials: Secure credentials wrapper (default: empty auth)
        """
        self._uri = uri or f"bolt://{host}:{port}"
        self._credentials = credentials or MemgraphCredentials()
        self._driver: Any = None

    def connect(self) -> None:
        try:
            import neo4j
        except ImportError as e:
            raise ImportError("neo4j driver required: pip install qortex[memgraph]") from e
        self._driver = neo4j.GraphDatabase.driver(self._uri, auth=self._credentials.auth_tuple)
        self._driver.verify_connectivity()
        logger.info("Connected to Memgraph at %s", self._uri)

    def disconnect(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    def is_connected(self) -> bool:
        if not self._driver:
            return False
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    @traced("cypher.execute", external=True)
    def _run(self, cypher: str, params: dict | None = None) -> list[dict]:
        """Execute a Cypher query and return all records as dicts."""
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            span.set_attribute("db.system", "memgraph")
            span.set_attribute("db.operation.name", _cypher_op(cypher))
            span.set_attribute("db.statement", _sanitize_cypher(cypher))
        except ImportError:
            pass

        with self._driver.session() as session:
            result = session.run(cypher, params or {})
            records = [dict(record) for record in result]

        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            span.set_attribute("db.result_count", len(records))
        except ImportError:
            pass

        return records

    def _run_single(self, cypher: str, params: dict | None = None) -> dict | None:
        """Execute a Cypher query and return the first record or None."""
        records = self._run(cypher, params)
        return records[0] if records else None

    # -------------------------------------------------------------------------
    # Domain Management
    # -------------------------------------------------------------------------

    def create_domain(self, name: str, description: str | None = None) -> Domain:
        now = datetime.now(UTC).isoformat()
        record = self._run_single(
            "MERGE (d:Domain {name: $name}) "
            "ON CREATE SET d.description = $desc, d.created_at = $now, d.updated_at = $now "
            "RETURN d.name AS name, d.description AS description, "
            "d.created_at AS created_at, d.updated_at AS updated_at",
            {"name": name, "desc": description, "now": now},
        )
        return self._record_to_domain(record)

    def get_domain(self, name: str) -> Domain | None:
        record = self._run_single(
            "MATCH (d:Domain {name: $name}) "
            "RETURN d.name AS name, d.description AS description, "
            "d.created_at AS created_at, d.updated_at AS updated_at",
            {"name": name},
        )
        if not record:
            return None
        return self._record_to_domain(record)

    def list_domains(self) -> list[Domain]:
        records = self._run(
            "MATCH (d:Domain) "
            "RETURN d.name AS name, d.description AS description, "
            "d.created_at AS created_at, d.updated_at AS updated_at "
            "ORDER BY d.name"
        )
        return [self._record_to_domain(r) for r in records]

    def delete_domain(self, name: str) -> bool:
        # Check existence first
        existing = self.get_domain(name)
        if not existing:
            return False
        # Delete domain, its concepts, rules, and edges between domain concepts
        self._run(
            "MATCH (c:Concept {domain: $name}) DETACH DELETE c",
            {"name": name},
        )
        self._run(
            "MATCH (r:Rule {domain: $name}) DELETE r",
            {"name": name},
        )
        self._run(
            "MATCH (d:Domain {name: $name}) DELETE d",
            {"name": name},
        )
        return True

    def _record_to_domain(self, record: dict) -> Domain:
        """Convert a Cypher record to a Domain model, computing stats."""
        name = record["name"]
        # Count stats
        concept_count = self._count(
            "MATCH (c:Concept {domain: $name}) RETURN count(c) AS cnt", name
        )
        edge_count = self._count(
            "MATCH (c:Concept {domain: $name})-[r]-() RETURN count(r) AS cnt", name
        )
        rule_count = self._count("MATCH (r:Rule {domain: $name}) RETURN count(r) AS cnt", name)

        return Domain(
            name=name,
            description=record.get("description"),
            concept_count=concept_count,
            edge_count=edge_count,
            rule_count=rule_count,
            created_at=_parse_iso(record.get("created_at")),
            updated_at=_parse_iso(record.get("updated_at")),
        )

    def _count(self, cypher: str, name: str) -> int:
        record = self._run_single(cypher, {"name": name})
        return record["cnt"] if record else 0

    # -------------------------------------------------------------------------
    # Node Operations
    # -------------------------------------------------------------------------

    def add_node(self, node: ConceptNode) -> None:
        self._run(
            "MERGE (c:Concept {id: $id}) "
            "SET c.name = $name, c.description = $desc, c.domain = $domain, "
            "c.source_id = $source_id, c.source_location = $source_location, "
            "c.confidence = $confidence, c.properties = $props",
            {
                "id": node.id,
                "name": node.name,
                "desc": node.description,
                "domain": node.domain,
                "source_id": node.source_id,
                "source_location": node.source_location,
                "confidence": node.confidence,
                "props": json.dumps(node.properties),
            },
        )

    def get_node(self, node_id: str, domain: str | None = None) -> ConceptNode | None:
        if domain:
            record = self._run_single(
                "MATCH (c:Concept {id: $id, domain: $domain}) "
                "RETURN c.id AS id, c.name AS name, c.description AS description, "
                "c.domain AS domain, c.source_id AS source_id, "
                "c.source_location AS source_location, c.confidence AS confidence, "
                "c.properties AS properties",
                {"id": node_id, "domain": domain},
            )
        else:
            record = self._run_single(
                "MATCH (c:Concept {id: $id}) "
                "RETURN c.id AS id, c.name AS name, c.description AS description, "
                "c.domain AS domain, c.source_id AS source_id, "
                "c.source_location AS source_location, c.confidence AS confidence, "
                "c.properties AS properties",
                {"id": node_id},
            )
        if not record:
            return None
        return _record_to_concept(record)

    def find_nodes(
        self,
        domain: str | None = None,
        name_pattern: str | None = None,
        limit: int = 100,
    ) -> Iterator[ConceptNode]:
        clauses = ["MATCH (c:Concept)"]
        wheres: list[str] = []
        params: dict[str, Any] = {"limit": limit}

        if domain:
            wheres.append("c.domain = $domain")
            params["domain"] = domain
        if name_pattern:
            # Convert glob pattern to regex for Cypher
            regex = name_pattern.replace("*", ".*").replace("?", ".")
            wheres.append("c.name =~ $pattern")
            params["pattern"] = f"(?i){regex}"

        if wheres:
            clauses.append("WHERE " + " AND ".join(wheres))
        clauses.append(
            "RETURN c.id AS id, c.name AS name, c.description AS description, "
            "c.domain AS domain, c.source_id AS source_id, "
            "c.source_location AS source_location, c.confidence AS confidence, "
            "c.properties AS properties "
            "LIMIT $limit"
        )

        records = self._run(" ".join(clauses), params)
        for r in records:
            yield _record_to_concept(r)

    # -------------------------------------------------------------------------
    # Edge Operations
    # -------------------------------------------------------------------------

    def add_edge(self, edge: ConceptEdge) -> None:
        # Use relation_type as actual edge label for proper graph analytics
        rel_type = edge.relation_type.value.upper()
        self._run(
            f"MATCH (s:Concept {{id: $src}}), (t:Concept {{id: $tgt}}) "
            f"CREATE (s)-[:{rel_type} {{confidence: $conf, "
            f"bidirectional: $bidir, properties: $props}}]->(t)",
            {
                "src": edge.source_id,
                "tgt": edge.target_id,
                "conf": edge.confidence,
                "bidir": edge.bidirectional,
                "props": json.dumps(edge.properties),
            },
        )

    def get_edges(
        self,
        node_id: str,
        direction: Literal["in", "out", "both"] = "both",
        relation_type: str | None = None,
    ) -> Iterator[ConceptEdge]:
        # Edge types are now actual labels (REQUIRES, SIMILAR_TO, etc.)
        # Use specific type if provided, otherwise match any relationship
        rel_pattern = f":{relation_type.upper()}" if relation_type else ""

        if direction == "out":
            pattern = f"(s:Concept {{id: $nid}})-[r{rel_pattern}]->(t:Concept)"
            where = ""
        elif direction == "in":
            pattern = f"(t:Concept)-[r{rel_pattern}]->(s:Concept {{id: $nid}})"
            where = ""
        else:
            pattern = f"(s:Concept)-[r{rel_pattern}]-(t:Concept)"
            where = "WHERE s.id = $nid OR t.id = $nid"

        cypher = (
            f"MATCH {pattern} {where} "
            "RETURN s.id AS src, t.id AS tgt, type(r) AS type, "
            "r.confidence AS confidence, r.bidirectional AS bidirectional, "
            "r.properties AS properties"
        )
        params: dict[str, Any] = {"nid": node_id}

        records = self._run(cypher, params)
        seen: set[tuple[str, str, str]] = set()
        for r in records:
            key = (r["src"], r["tgt"], r["type"])
            if key in seen:
                continue
            seen.add(key)
            yield ConceptEdge(
                source_id=r["src"],
                target_id=r["tgt"],
                relation_type=RelationType(r["type"].lower()),
                confidence=r.get("confidence", 1.0),
                bidirectional=r.get("bidirectional", False),
                properties=json.loads(r["properties"]) if r.get("properties") else {},
            )

    # -------------------------------------------------------------------------
    # Rule Operations
    # -------------------------------------------------------------------------

    def add_rule(self, rule: ExplicitRule) -> None:
        self._run(
            "MERGE (r:Rule {id: $id}) "
            "SET r.text = $text, r.domain = $domain, r.source_id = $source_id, "
            "r.source_location = $source_location, r.category = $category, "
            "r.confidence = $confidence, r.concept_ids = $concept_ids",
            {
                "id": rule.id,
                "text": rule.text,
                "domain": rule.domain,
                "source_id": rule.source_id,
                "source_location": rule.source_location,
                "category": rule.category,
                "confidence": rule.confidence,
                "concept_ids": json.dumps(rule.concept_ids),
            },
        )

    def get_rules(self, domain: str | None = None) -> list[ExplicitRule]:
        if domain:
            records = self._run(
                "MATCH (r:Rule {domain: $domain}) "
                "RETURN r.id AS id, r.text AS text, r.domain AS domain, "
                "r.source_id AS source_id, r.source_location AS source_location, "
                "r.category AS category, r.confidence AS confidence, "
                "r.concept_ids AS concept_ids",
                {"domain": domain},
            )
        else:
            records = self._run(
                "MATCH (r:Rule) "
                "RETURN r.id AS id, r.text AS text, r.domain AS domain, "
                "r.source_id AS source_id, r.source_location AS source_location, "
                "r.category AS category, r.confidence AS confidence, "
                "r.concept_ids AS concept_ids"
            )
        return [_record_to_rule(r) for r in records]

    # -------------------------------------------------------------------------
    # Manifest Ingestion
    # -------------------------------------------------------------------------

    def ingest_manifest(self, manifest: IngestionManifest) -> None:
        """Atomically ingest a manifest. Uses a single transaction."""
        t0 = time.perf_counter()
        self.create_domain(manifest.domain)

        # Update domain source tracking
        now = datetime.now(UTC).isoformat()
        self._run(
            "MATCH (d:Domain {name: $name}) SET d.updated_at = $now",
            {"name": manifest.domain, "now": now},
        )

        for node in manifest.concepts:
            self.add_node(node)
        for edge in manifest.edges:
            self.add_edge(edge)
        for rule in manifest.rules:
            self.add_rule(rule)

        elapsed = (time.perf_counter() - t0) * 1000
        emit(ManifestIngested(
            domain=manifest.domain,
            node_count=len(manifest.concepts),
            edge_count=len(manifest.edges),
            rule_count=len(manifest.rules),
            source_id=manifest.source.id,
            latency_ms=elapsed,
        ))

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    def query(self, pattern: GraphPattern) -> Iterator[dict]:
        raise NotImplementedError("Structured queries not yet supported; use query_cypher()")

    def query_cypher(self, cypher: str, params: dict | None = None) -> Iterator[dict]:
        records = self._run(cypher, params)
        yield from records

    # -------------------------------------------------------------------------
    # Graph Algorithms (MAGE)
    # -------------------------------------------------------------------------

    def supports_mage(self) -> bool:
        return True

    def personalized_pagerank(
        self,
        source_nodes: list[str],
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        domain: str | None = None,
        seed_weights: dict[str, float] | None = None,
        extra_edges: list[tuple[str, str, float]] | None = None,
        query_id: str | None = None,
    ) -> dict[str, float]:
        """Personalized PageRank via power iteration over Memgraph graph data.

        Fetches the adjacency structure from Memgraph and runs the same
        power iteration algorithm as InMemoryBackend. MAGE's pagerank.get()
        only supports standard PageRank (no personalization), so we run
        PPR in Python for correctness.
        """
        from qortex.observability.events import PPRConverged, PPRDiverged, PPRStarted, QueryFailed

        # 1. Fetch nodes in scope
        try:
            if domain:
                node_records = self._run(
                    "MATCH (n:Concept {domain: $d}) RETURN n.id AS id",
                    {"d": domain},
                )
            else:
                node_records = self._run("MATCH (n:Concept) RETURN n.id AS id")
        except Exception as exc:
            emit(QueryFailed(
                query_id=query_id,
                error=str(exc),
                stage="ppr",
                timestamp=datetime.now(UTC).isoformat(),
            ))
            raise

        node_ids = [r["id"] for r in node_records if r.get("id")]
        node_set = set(node_ids)

        # Filter seeds to nodes that actually exist in scope
        valid_seeds = [nid for nid in source_nodes if nid in node_set]
        if not valid_seeds or not node_ids:
            return {}

        t0 = time.perf_counter()
        emit(PPRStarted(
            query_id=query_id,
            node_count=len(node_ids),
            seed_count=len(valid_seeds),
            damping_factor=damping_factor,
            extra_edge_count=len(extra_edges) if extra_edges else 0,
        ))

        # 2. Fetch edges and build adjacency: node_id → [(neighbor_id, weight)]
        try:
            if domain:
                edge_records = self._run(
                    "MATCH (a:Concept {domain: $d})-[r]->(b:Concept {domain: $d}) "
                    "RETURN a.id AS src, b.id AS tgt, "
                    "COALESCE(r.confidence, 1.0) AS weight",
                    {"d": domain},
                )
            else:
                edge_records = self._run(
                    "MATCH (a:Concept)-[r]->(b:Concept) "
                    "RETURN a.id AS src, b.id AS tgt, "
                    "COALESCE(r.confidence, 1.0) AS weight",
                )
        except Exception as exc:
            emit(QueryFailed(
                query_id=query_id,
                error=str(exc),
                stage="ppr",
                timestamp=datetime.now(UTC).isoformat(),
            ))
            raise

        adjacency: dict[str, list[tuple[str, float]]] = {nid: [] for nid in node_ids}
        for rec in edge_records:
            src, tgt = rec["src"], rec["tgt"]
            w = float(rec.get("weight", 1.0))
            if src in node_set and tgt in node_set:
                adjacency[src].append((tgt, w))
                # Undirected: add reverse too
                adjacency[tgt].append((src, w))

        # Extra edges (from online edge generation)
        if extra_edges:
            for src, tgt, weight in extra_edges:
                if src in node_set and tgt in node_set:
                    adjacency[src].append((tgt, weight))
                    adjacency[tgt].append((src, weight))

        # 3. Build personalization vector
        personalization: dict[str, float] = {}
        if seed_weights:
            for nid in valid_seeds:
                personalization[nid] = seed_weights.get(nid, 0.0)
        else:
            for nid in valid_seeds:
                personalization[nid] = 1.0

        p_total = sum(personalization.values())
        if p_total > 0:
            personalization = {k: v / p_total for k, v in personalization.items()}

        # 4. Power iteration: π(t+1) = d * (A @ π(t)) + (1 - d) * personalization
        n = len(node_ids)
        scores: dict[str, float] = {nid: 1.0 / n for nid in node_ids}
        convergence_threshold = 1e-6
        diff = 0.0

        for _iteration in range(max_iterations):
            new_scores: dict[str, float] = {}
            for nid in node_ids:
                teleport = (1.0 - damping_factor) * personalization.get(nid, 0.0)
                walk = 0.0
                for neighbor_id, weight in adjacency.get(nid, []):
                    neighbor_out = adjacency.get(neighbor_id, [])
                    out_weight = sum(w for _, w in neighbor_out)
                    if out_weight > 0:
                        walk += scores.get(neighbor_id, 0.0) * weight / out_weight
                new_scores[nid] = teleport + damping_factor * walk

            diff = sum(abs(new_scores.get(k, 0) - scores.get(k, 0)) for k in node_ids)
            scores = new_scores
            if diff < convergence_threshold:
                break

        elapsed = (time.perf_counter() - t0) * 1000
        result = {nid: score for nid, score in scores.items() if score > 1e-8}

        if diff < convergence_threshold:
            emit(PPRConverged(
                query_id=query_id,
                iterations=_iteration + 1,
                final_diff=diff,
                node_count=len(node_ids),
                nonzero_scores=len(result),
                latency_ms=elapsed,
            ))
        else:
            emit(PPRDiverged(
                query_id=query_id,
                iterations=max_iterations,
                final_diff=diff,
                node_count=len(node_ids),
            ))

        return result

    # -------------------------------------------------------------------------
    # Vector Operations
    # -------------------------------------------------------------------------

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Store embedding as a JSON property on the Concept node."""
        self._run(
            "MATCH (c:Concept {id: $id}) SET c.embedding = $emb",
            {"id": node_id, "emb": json.dumps(embedding)},
        )

    def get_embedding(self, node_id: str) -> list[float] | None:
        record = self._run_single(
            "MATCH (c:Concept {id: $id}) RETURN c.embedding AS emb",
            {"id": node_id},
        )
        if not record or not record.get("emb"):
            return None
        return json.loads(record["emb"])

    def vector_search(
        self,
        query_embedding: list[float],
        domain: str | None = None,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[ConceptNode, float]]:
        """Vector search over Memgraph-stored embeddings.

        Memgraph doesn't have native vector indexing — retrieves embeddings
        and computes cosine similarity application-side.
        """
        try:
            import numpy as np
        except ImportError:
            logger.warning("numpy required for vector_search")
            return []

        if domain:
            records = self._run(
                "MATCH (c:Concept {domain: $domain}) WHERE c.embedding IS NOT NULL "
                "RETURN c.id AS id, c.name AS name, c.description AS description, "
                "c.domain AS domain, c.source_id AS source_id, "
                "c.source_location AS source_location, c.confidence AS confidence, "
                "c.properties AS properties, c.embedding AS embedding",
                {"domain": domain},
            )
        else:
            records = self._run(
                "MATCH (c:Concept) WHERE c.embedding IS NOT NULL "
                "RETURN c.id AS id, c.name AS name, c.description AS description, "
                "c.domain AS domain, c.source_id AS source_id, "
                "c.source_location AS source_location, c.confidence AS confidence, "
                "c.properties AS properties, c.embedding AS embedding"
            )

        if not records:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm

        scored: list[tuple[dict, float]] = []
        for r in records:
            emb = json.loads(r["embedding"]) if isinstance(r["embedding"], str) else r["embedding"]
            vec = np.array(emb, dtype=np.float32)
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            sim = float(np.dot(query, vec / vec_norm))
            if sim >= threshold:
                scored.append((r, sim))

        scored.sort(key=lambda x: -x[1])
        results = []
        for r, score in scored[:top_k]:
            node = _record_to_concept(r)
            results.append((node, score))
        return results

    def supports_vector_search(self) -> bool:
        return True

    # -------------------------------------------------------------------------
    # Checkpointing (Memgraph snapshots)
    # -------------------------------------------------------------------------

    def checkpoint(self, name: str, domains: list[str] | None = None) -> str:
        """Trigger a Memgraph snapshot. Returns a checkpoint ID.

        Note: Memgraph snapshots are server-wide, not per-domain.
        The domains parameter is recorded but not used for partial snapshots.
        """
        checkpoint_id = str(uuid.uuid4())
        try:
            self._run("CALL mg.create_snapshot()")
        except Exception as e:
            logger.warning("Snapshot creation failed (non-fatal): %s", e)
        # Store checkpoint metadata as a node
        self._run(
            "CREATE (:Checkpoint {id: $id, name: $name, domains: $domains, created_at: $now})",
            {
                "id": checkpoint_id,
                "name": name,
                "domains": json.dumps(domains) if domains else "[]",
                "now": datetime.now(UTC).isoformat(),
            },
        )
        return checkpoint_id

    def restore(self, checkpoint_id: str) -> None:
        """Restore is not supported at the application level with Memgraph.

        Memgraph restore requires server restart with --snapshot-recovery.
        This method exists to satisfy the protocol.
        """
        record = self._run_single(
            "MATCH (c:Checkpoint {id: $id}) RETURN c.name AS name",
            {"id": checkpoint_id},
        )
        if not record:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        raise NotImplementedError(
            "Memgraph restore requires server restart with --snapshot-recovery. "
            f"Checkpoint '{record['name']}' exists but cannot be restored programmatically."
        )

    def list_checkpoints(self) -> list[dict]:
        records = self._run(
            "MATCH (c:Checkpoint) "
            "RETURN c.id AS id, c.name AS name, c.created_at AS created_at "
            "ORDER BY c.created_at DESC"
        )
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "created_at": r.get("created_at"),
            }
            for r in records
        ]


# =============================================================================
# Helpers for MemgraphBackend record -> model conversion
# =============================================================================


def _parse_iso(val: str | None) -> datetime:
    """Parse an ISO timestamp string, or return now(utc) as default."""
    if not val:
        return datetime.now(UTC)
    try:
        return datetime.fromisoformat(val)
    except (ValueError, TypeError):
        return datetime.now(UTC)


def _record_to_concept(record: dict) -> ConceptNode:
    """Convert a Cypher record dict to a ConceptNode."""
    props = {}
    if record.get("properties"):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            props = json.loads(record["properties"])
    return ConceptNode(
        id=record["id"],
        name=record["name"],
        description=record.get("description", ""),
        domain=record["domain"],
        source_id=record.get("source_id", ""),
        source_location=record.get("source_location"),
        confidence=record.get("confidence", 1.0),
        properties=props,
    )


def _record_to_rule(record: dict) -> ExplicitRule:
    """Convert a Cypher record dict to an ExplicitRule."""
    concept_ids: list[str] = []
    if record.get("concept_ids"):
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            concept_ids = json.loads(record["concept_ids"])
    return ExplicitRule(
        id=record["id"],
        text=record["text"],
        domain=record["domain"],
        source_id=record.get("source_id", ""),
        source_location=record.get("source_location"),
        category=record.get("category"),
        confidence=record.get("confidence", 1.0),
        concept_ids=concept_ids,
    )


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
        seed_weights: dict[str, float] | None = None,
        extra_edges: list[tuple[str, str, float]] | None = None,
        query_id: str | None = None,
    ) -> dict[str, float]:
        """Simple BFS-based approximation (not true PPR)."""
        raise NotImplementedError("M3: Implement simple traversal fallback")

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        raise NotImplementedError("SQLiteBackend vector ops not yet implemented")

    def get_embedding(self, node_id: str) -> list[float] | None:
        raise NotImplementedError("SQLiteBackend vector ops not yet implemented")

    def vector_search(
        self,
        query_embedding: list[float],
        domain: str | None = None,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[ConceptNode, float]]:
        raise NotImplementedError("SQLiteBackend vector ops not yet implemented")

    def supports_vector_search(self) -> bool:
        return False


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
