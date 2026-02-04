"""Memgraph backend implementation using gqlalchemy.

Primary backend for qortex - provides:
- Native graph operations
- MAGE algorithms (PPR, community detection)
- Domain isolation via labels
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Iterator, Literal

from gqlalchemy import Memgraph
from gqlalchemy.query_builders.memgraph_query_builder import QueryBuilder

from .backend import GraphBackend, GraphPattern
from .models import (
    ConceptEdge,
    ConceptNode,
    Domain,
    ExplicitRule,
    IngestionManifest,
    RelationType,
)


class MemgraphBackend(GraphBackend):
    """Memgraph implementation of GraphBackend.

    Domain isolation: Each domain is represented by a label on nodes.
    Nodes have labels: [:Concept:DomainName]
    Edges are typed by RelationType.

    Checkpointing: Uses Memgraph's snapshot mechanism or manual export.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 7687,
        username: str = "",
        password: str = "",
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self._db: Memgraph | None = None

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------

    def connect(self) -> None:
        """Establish connection to Memgraph."""
        self._db = Memgraph(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
        )
        # Test connection
        self._db.execute("RETURN 1")

    def disconnect(self) -> None:
        """Close connection."""
        self._db = None

    def is_connected(self) -> bool:
        """Check if connected to Memgraph."""
        if self._db is None:
            return False
        try:
            self._db.execute("RETURN 1")
            return True
        except Exception:
            return False

    @property
    def db(self) -> Memgraph:
        """Get database connection, raising if not connected."""
        if self._db is None:
            raise RuntimeError("Not connected to Memgraph. Call connect() first.")
        return self._db

    # -------------------------------------------------------------------------
    # Domain Management
    # -------------------------------------------------------------------------

    def create_domain(self, name: str, description: str | None = None) -> Domain:
        """Create a new domain.

        Domains are virtual - they're labels on nodes, not separate databases.
        We track domain metadata in a special :Domain node.
        """
        # Check if domain already exists
        result = list(self.db.execute_and_fetch(
            f"MATCH (d:Domain {{name: '{name}'}}) RETURN d"
        ))
        if result:
            raise ValueError(f"Domain '{name}' already exists")

        # Create domain metadata node
        now = datetime.utcnow().isoformat()
        self.db.execute(f"""
            CREATE (d:Domain {{
                name: '{name}',
                description: '{description or ''}',
                created_at: '{now}',
                updated_at: '{now}',
                concept_count: 0,
                edge_count: 0,
                rule_count: 0
            }})
        """)

        return Domain(
            name=name,
            description=description,
            created_at=datetime.fromisoformat(now),
            updated_at=datetime.fromisoformat(now),
        )

    def get_domain(self, name: str) -> Domain | None:
        """Get domain by name."""
        result = list(self.db.execute_and_fetch(
            f"MATCH (d:Domain {{name: '{name}'}}) RETURN d"
        ))
        if not result:
            return None

        d = result[0]["d"]
        return Domain(
            name=d["name"],
            description=d.get("description"),
            concept_count=d.get("concept_count", 0),
            edge_count=d.get("edge_count", 0),
            rule_count=d.get("rule_count", 0),
            source_ids=d.get("source_ids", []),
            created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.utcnow(),
            updated_at=datetime.fromisoformat(d["updated_at"]) if d.get("updated_at") else datetime.utcnow(),
        )

    def list_domains(self) -> list[Domain]:
        """List all domains."""
        result = self.db.execute_and_fetch("MATCH (d:Domain) RETURN d")
        domains = []
        for row in result:
            d = row["d"]
            domains.append(Domain(
                name=d["name"],
                description=d.get("description"),
                concept_count=d.get("concept_count", 0),
                edge_count=d.get("edge_count", 0),
                rule_count=d.get("rule_count", 0),
            ))
        return domains

    def delete_domain(self, name: str) -> bool:
        """Delete domain and all its contents."""
        # Delete all nodes with domain label
        self.db.execute(f"MATCH (n:{name}) DETACH DELETE n")
        # Delete domain metadata
        result = self.db.execute(f"MATCH (d:Domain {{name: '{name}'}}) DELETE d")
        return True

    def _update_domain_stats(self, domain_name: str) -> None:
        """Update domain statistics."""
        # Count concepts
        concept_count = list(self.db.execute_and_fetch(
            f"MATCH (n:Concept:{domain_name}) RETURN count(n) as count"
        ))[0]["count"]

        # Count edges (relationships between concepts in this domain)
        edge_count = list(self.db.execute_and_fetch(
            f"MATCH (a:Concept:{domain_name})-[r]->(b:Concept:{domain_name}) RETURN count(r) as count"
        ))[0]["count"]

        # Count rules
        rule_count = list(self.db.execute_and_fetch(
            f"MATCH (r:Rule:{domain_name}) RETURN count(r) as count"
        ))[0]["count"]

        # Update domain node
        now = datetime.utcnow().isoformat()
        self.db.execute(f"""
            MATCH (d:Domain {{name: '{domain_name}'}})
            SET d.concept_count = {concept_count},
                d.edge_count = {edge_count},
                d.rule_count = {rule_count},
                d.updated_at = '{now}'
        """)

    # -------------------------------------------------------------------------
    # Node Operations
    # -------------------------------------------------------------------------

    def add_node(self, node: ConceptNode) -> None:
        """Add a concept node to its domain."""
        # Ensure domain exists
        if not self.get_domain(node.domain):
            self.create_domain(node.domain)

        # Escape strings for Cypher
        props = {
            "id": node.id,
            "name": node.name,
            "description": node.description,
            "source_id": node.source_id,
            "source_location": node.source_location or "",
            "confidence": node.confidence,
            "properties": json.dumps(node.properties),
        }

        # Create node with Concept label + domain label
        self.db.execute(f"""
            CREATE (n:Concept:{node.domain} {{
                id: $id,
                name: $name,
                description: $description,
                source_id: $source_id,
                source_location: $source_location,
                confidence: $confidence,
                properties: $properties
            }})
        """, props)

    def get_node(self, node_id: str, domain: str | None = None) -> ConceptNode | None:
        """Get node by ID."""
        if domain:
            query = f"MATCH (n:Concept:{domain} {{id: $id}}) RETURN n"
        else:
            query = "MATCH (n:Concept {id: $id}) RETURN n"

        result = list(self.db.execute_and_fetch(query, {"id": node_id}))
        if not result:
            return None

        n = result[0]["n"]
        return self._node_from_record(n, domain or self._extract_domain_from_id(node_id))

    def find_nodes(
        self,
        domain: str | None = None,
        name_pattern: str | None = None,
        limit: int = 100,
    ) -> Iterator[ConceptNode]:
        """Find nodes matching criteria."""
        if domain:
            query = f"MATCH (n:Concept:{domain})"
        else:
            query = "MATCH (n:Concept)"

        if name_pattern:
            query += f" WHERE toLower(n.name) CONTAINS toLower('{name_pattern}')"

        query += f" RETURN n LIMIT {limit}"

        for row in self.db.execute_and_fetch(query):
            n = row["n"]
            # Extract domain from labels
            node_domain = domain or self._extract_domain_from_labels(n)
            yield self._node_from_record(n, node_domain)

    def _node_from_record(self, record: dict[str, Any], domain: str) -> ConceptNode:
        """Convert a Memgraph record to ConceptNode."""
        return ConceptNode(
            id=record["id"],
            name=record["name"],
            description=record.get("description", ""),
            domain=domain,
            source_id=record.get("source_id", ""),
            source_location=record.get("source_location"),
            confidence=record.get("confidence", 1.0),
            properties=json.loads(record.get("properties", "{}")),
        )

    def _extract_domain_from_id(self, node_id: str) -> str:
        """Extract domain from node ID (format: domain:name)."""
        return node_id.split(":")[0] if ":" in node_id else "default"

    def _extract_domain_from_labels(self, record: Any) -> str:
        """Extract domain from node labels."""
        # This is a simplification - in practice we'd inspect labels
        return "default"

    # -------------------------------------------------------------------------
    # Edge Operations
    # -------------------------------------------------------------------------

    def add_edge(self, edge: ConceptEdge) -> None:
        """Add an edge between concepts."""
        rel_type = edge.relation_type.value.upper()
        props = {
            "confidence": edge.confidence,
            "properties": json.dumps(edge.properties),
        }

        query = f"""
            MATCH (a:Concept {{id: $source_id}})
            MATCH (b:Concept {{id: $target_id}})
            CREATE (a)-[r:{rel_type} {{
                confidence: $confidence,
                properties: $properties
            }}]->(b)
        """

        self.db.execute(query, {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            **props,
        })

        # If bidirectional, create reverse edge
        if edge.bidirectional:
            self.db.execute(f"""
                MATCH (a:Concept {{id: $source_id}})
                MATCH (b:Concept {{id: $target_id}})
                CREATE (b)-[r:{rel_type} {{
                    confidence: $confidence,
                    properties: $properties
                }}]->(a)
            """, {
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                **props,
            })

    def get_edges(
        self,
        node_id: str,
        direction: Literal["in", "out", "both"] = "both",
        relation_type: str | None = None,
    ) -> Iterator[ConceptEdge]:
        """Get edges connected to a node."""
        if direction == "out":
            pattern = "(a {id: $id})-[r]->(b)"
        elif direction == "in":
            pattern = "(a)-[r]->(b {id: $id})"
        else:
            pattern = "(a {id: $id})-[r]-(b)"

        query = f"MATCH {pattern} RETURN a.id as source, b.id as target, type(r) as rel_type, r"

        for row in self.db.execute_and_fetch(query, {"id": node_id}):
            try:
                rel_type = RelationType(row["rel_type"].lower())
            except ValueError:
                rel_type = RelationType.SIMILAR_TO  # fallback

            if relation_type and rel_type.value != relation_type:
                continue

            yield ConceptEdge(
                source_id=row["source"],
                target_id=row["target"],
                relation_type=rel_type,
                confidence=row["r"].get("confidence", 1.0),
                properties=json.loads(row["r"].get("properties", "{}")),
            )

    # -------------------------------------------------------------------------
    # Manifest Ingestion
    # -------------------------------------------------------------------------

    def ingest_manifest(self, manifest: IngestionManifest) -> None:
        """Atomically ingest a full manifest into the KG."""
        # Ensure domain exists
        if not self.get_domain(manifest.domain):
            self.create_domain(manifest.domain, f"Ingested from {manifest.source.name}")

        # Add all concepts
        for concept in manifest.concepts:
            self.add_node(concept)

        # Add all edges
        for edge in manifest.edges:
            self.add_edge(edge)

        # Add all explicit rules as Rule nodes
        for rule in manifest.rules:
            self._add_rule(rule)

        # Update domain stats
        self._update_domain_stats(manifest.domain)

    def _add_rule(self, rule: ExplicitRule) -> None:
        """Add an explicit rule to the graph."""
        props = {
            "id": rule.id,
            "text": rule.text,
            "source_id": rule.source_id,
            "source_location": rule.source_location or "",
            "category": rule.category or "",
            "confidence": rule.confidence,
            "concept_ids": json.dumps(rule.concept_ids),
        }

        self.db.execute(f"""
            CREATE (r:Rule:{rule.domain} {{
                id: $id,
                text: $text,
                source_id: $source_id,
                source_location: $source_location,
                category: $category,
                confidence: $confidence,
                concept_ids: $concept_ids
            }})
        """, props)

        # Link rule to its concepts
        for concept_id in rule.concept_ids:
            self.db.execute(f"""
                MATCH (r:Rule {{id: $rule_id}})
                MATCH (c:Concept {{id: $concept_id}})
                CREATE (r)-[:OPERATIONALIZES]->(c)
            """, {"rule_id": rule.id, "concept_id": concept_id})

    # -------------------------------------------------------------------------
    # Query
    # -------------------------------------------------------------------------

    def query(self, pattern: GraphPattern) -> Iterator[dict]:
        """Execute a structured query."""
        # TODO: Implement structured query DSL
        raise NotImplementedError("Structured queries not yet implemented")

    def query_cypher(self, cypher: str, params: dict | None = None) -> Iterator[dict]:
        """Execute raw Cypher query."""
        for row in self.db.execute_and_fetch(cypher, params or {}):
            yield dict(row)

    # -------------------------------------------------------------------------
    # Graph Algorithms
    # -------------------------------------------------------------------------

    def supports_mage(self) -> bool:
        """Memgraph supports MAGE algorithms."""
        return True

    def personalized_pagerank(
        self,
        source_nodes: list[str],
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        domain: str | None = None,
    ) -> dict[str, float]:
        """Personalized PageRank from source nodes using MAGE.

        Uses pagerank_online.set / pagerank_online.get or pagerank.get
        """
        if not source_nodes:
            return {}

        # Build subgraph query if domain specified
        if domain:
            subgraph = f"MATCH (n:Concept:{domain})-[r]->(m:Concept:{domain})"
        else:
            subgraph = "MATCH (n:Concept)-[r]->(m:Concept)"

        # Get source node internal IDs
        source_ids_str = ", ".join(f"'{sid}'" for sid in source_nodes)

        try:
            # Try MAGE pagerank with personalization
            # Note: MAGE API may vary - this is a common pattern
            result = self.db.execute_and_fetch(f"""
                {subgraph}
                WITH collect(n) + collect(m) as nodes, collect(r) as rels
                CALL pagerank.get(nodes, rels, {{
                    dampingFactor: {damping_factor},
                    maxIterations: {max_iterations}
                }})
                YIELD node, rank
                MATCH (node) WHERE node.id IS NOT NULL
                RETURN node.id as id, rank
            """)

            scores = {row["id"]: row["rank"] for row in result}

            # Boost source nodes (personalization)
            for sid in source_nodes:
                if sid in scores:
                    scores[sid] *= 2.0  # Simple boost
                else:
                    scores[sid] = 1.0

            return scores

        except Exception as e:
            # Fallback to simple BFS if MAGE not available
            return self._bfs_scores(source_nodes, domain)

    def _bfs_scores(
        self,
        source_nodes: list[str],
        domain: str | None,
        max_depth: int = 3,
    ) -> dict[str, float]:
        """Simple BFS-based scoring as fallback."""
        scores: dict[str, float] = {}
        visited: set[str] = set()

        frontier = [(nid, 1.0) for nid in source_nodes]

        for depth in range(max_depth + 1):
            next_frontier = []
            decay = 0.5 ** depth

            for node_id, base_score in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)

                score = base_score * decay
                scores[node_id] = max(scores.get(node_id, 0), score)

                # Get neighbors
                for edge in self.get_edges(node_id, direction="both"):
                    neighbor = edge.target_id if edge.source_id == node_id else edge.source_id
                    if neighbor not in visited:
                        next_frontier.append((neighbor, score * edge.confidence))

            frontier = next_frontier

        return scores

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def checkpoint(self, name: str, domains: list[str] | None = None) -> str:
        """Create a named checkpoint.

        For now, uses a simple export-based approach.
        Could use Memgraph's snapshot mechanism in production.
        """
        checkpoint_id = hashlib.sha256(
            f"{name}:{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:12]

        # Export current state as checkpoint metadata
        self.db.execute(f"""
            CREATE (c:Checkpoint {{
                id: '{checkpoint_id}',
                name: '{name}',
                timestamp: '{datetime.utcnow().isoformat()}',
                domains: '{json.dumps(domains or [])}'
            }})
        """)

        # TODO: Actually snapshot the data (export to file or use Memgraph snapshots)

        return checkpoint_id

    def restore(self, checkpoint_id: str) -> None:
        """Restore to a checkpoint state."""
        # TODO: Implement actual restore
        raise NotImplementedError("Checkpoint restore not yet implemented")

    def list_checkpoints(self) -> list[dict]:
        """List available checkpoints."""
        result = self.db.execute_and_fetch("MATCH (c:Checkpoint) RETURN c ORDER BY c.timestamp DESC")
        return [
            {
                "id": row["c"]["id"],
                "name": row["c"]["name"],
                "timestamp": datetime.fromisoformat(row["c"]["timestamp"]),
                "domains": json.loads(row["c"].get("domains", "[]")),
            }
            for row in result
        ]
