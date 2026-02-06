"""In-memory graph backend for testing without Docker/Memgraph."""

from __future__ import annotations

import copy
import fnmatch
import uuid
from datetime import datetime, timezone
from typing import Iterator, Literal

from .backend import GraphBackend, GraphPattern
from .models import ConceptEdge, ConceptNode, Domain, ExplicitRule, IngestionManifest


class InMemoryBackend:
    """Full GraphBackend implementation using Python dicts.

    Designed for testing — no external dependencies required.
    Supports all GraphBackend operations except Cypher queries.
    """

    def __init__(self) -> None:
        self._connected = False
        self._domains: dict[str, Domain] = {}
        self._nodes: dict[str, ConceptNode] = {}
        self._edges: list[ConceptEdge] = []
        self._rules: dict[str, ExplicitRule] = {}
        self._checkpoints: dict[str, dict] = {}

    # -------------------------------------------------------------------------
    # Connection
    # -------------------------------------------------------------------------

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # -------------------------------------------------------------------------
    # Domain Management
    # -------------------------------------------------------------------------

    def create_domain(self, name: str, description: str | None = None) -> Domain:
        if name in self._domains:
            return self._domains[name]
        domain = Domain(
            name=name,
            description=description,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        self._domains[name] = domain
        return domain

    def get_domain(self, name: str) -> Domain | None:
        return self._domains.get(name)

    def list_domains(self) -> list[Domain]:
        return list(self._domains.values())

    def delete_domain(self, name: str) -> bool:
        if name not in self._domains:
            return False
        del self._domains[name]
        # Cascade: remove nodes in this domain
        node_ids_to_remove = {
            nid for nid, node in self._nodes.items() if node.domain == name
        }
        for nid in node_ids_to_remove:
            del self._nodes[nid]
        # Cascade: remove edges referencing removed nodes
        self._edges = [
            e for e in self._edges
            if e.source_id not in node_ids_to_remove
            and e.target_id not in node_ids_to_remove
        ]
        # Cascade: remove rules in this domain
        rule_ids_to_remove = {
            rid for rid, rule in self._rules.items() if rule.domain == name
        }
        for rid in rule_ids_to_remove:
            del self._rules[rid]
        return True

    # -------------------------------------------------------------------------
    # Node Operations
    # -------------------------------------------------------------------------

    def add_node(self, node: ConceptNode) -> None:
        self._nodes[node.id] = node
        # Update domain stats
        if node.domain in self._domains:
            self._recount_domain(node.domain)

    def get_node(self, node_id: str, domain: str | None = None) -> ConceptNode | None:
        node = self._nodes.get(node_id)
        if node and domain and node.domain != domain:
            return None
        return node

    def find_nodes(
        self,
        domain: str | None = None,
        name_pattern: str | None = None,
        limit: int = 100,
    ) -> Iterator[ConceptNode]:
        count = 0
        for node in self._nodes.values():
            if count >= limit:
                break
            if domain and node.domain != domain:
                continue
            if name_pattern and not fnmatch.fnmatch(node.name.lower(), name_pattern.lower()):
                continue
            yield node
            count += 1

    # -------------------------------------------------------------------------
    # Edge Operations
    # -------------------------------------------------------------------------

    def add_edge(self, edge: ConceptEdge) -> None:
        self._edges.append(edge)
        # Update domain stats for source node's domain
        source_node = self._nodes.get(edge.source_id)
        if source_node and source_node.domain in self._domains:
            self._recount_domain(source_node.domain)

    def get_edges(
        self,
        node_id: str,
        direction: Literal["in", "out", "both"] = "both",
        relation_type: str | None = None,
    ) -> Iterator[ConceptEdge]:
        for edge in self._edges:
            if relation_type and edge.relation_type.value != relation_type:
                continue
            if direction == "out" and edge.source_id == node_id:
                yield edge
            elif direction == "in" and edge.target_id == node_id:
                yield edge
            elif direction == "both" and (edge.source_id == node_id or edge.target_id == node_id):
                yield edge

    # -------------------------------------------------------------------------
    # Rule Operations (for explicit rules stored in graph)
    # -------------------------------------------------------------------------

    def add_rule(self, rule: ExplicitRule) -> None:
        self._rules[rule.id] = rule
        if rule.domain in self._domains:
            self._recount_domain(rule.domain)

    def get_rules(self, domain: str | None = None) -> list[ExplicitRule]:
        if domain:
            return [r for r in self._rules.values() if r.domain == domain]
        return list(self._rules.values())

    # -------------------------------------------------------------------------
    # Manifest Ingestion
    # -------------------------------------------------------------------------

    def ingest_manifest(self, manifest: IngestionManifest) -> None:
        # Auto-create domain if it doesn't exist
        if manifest.domain not in self._domains:
            self.create_domain(manifest.domain)

        # Track source
        domain = self._domains[manifest.domain]
        if manifest.source.id not in domain.source_ids:
            domain.source_ids.append(manifest.source.id)

        # Ingest nodes
        for node in manifest.concepts:
            self.add_node(node)

        # Ingest edges
        for edge in manifest.edges:
            self.add_edge(edge)

        # Ingest explicit rules
        for rule in manifest.rules:
            self.add_rule(rule)

        # Update domain stats and timestamp
        self._recount_domain(manifest.domain)
        domain.updated_at = datetime.now(timezone.utc)

    # -------------------------------------------------------------------------
    # Query (limited — no Cypher support)
    # -------------------------------------------------------------------------

    def query(self, pattern: GraphPattern) -> Iterator[dict]:
        raise NotImplementedError("InMemoryBackend does not support structured queries")

    def query_cypher(self, cypher: str, params: dict | None = None) -> Iterator[dict]:
        raise NotImplementedError("InMemoryBackend does not support Cypher queries")

    # -------------------------------------------------------------------------
    # Graph Algorithms
    # -------------------------------------------------------------------------

    def supports_mage(self) -> bool:
        return False

    def personalized_pagerank(
        self,
        source_nodes: list[str],
        damping_factor: float = 0.85,
        max_iterations: int = 100,
        domain: str | None = None,
    ) -> dict[str, float]:
        """BFS-based approximation of PPR.

        2-hop traversal with exponential decay by depth.
        Not true PPR, but useful for testing the retrieval pipeline.
        """
        scores: dict[str, float] = {}
        # Seed nodes get score 1.0
        for nid in source_nodes:
            if nid in self._nodes:
                if domain and self._nodes[nid].domain != domain:
                    continue
                scores[nid] = 1.0

        # BFS expansion
        frontier = list(scores.keys())
        for depth in range(1, 3):  # 2-hop
            decay = damping_factor ** depth
            next_frontier: list[str] = []
            for nid in frontier:
                for edge in self._edges:
                    neighbor = None
                    if edge.source_id == nid:
                        neighbor = edge.target_id
                    elif edge.target_id == nid:
                        neighbor = edge.source_id

                    if neighbor and neighbor in self._nodes:
                        if domain and self._nodes[neighbor].domain != domain:
                            continue
                        score = decay * edge.confidence
                        if neighbor not in scores or scores[neighbor] < score:
                            scores[neighbor] = score
                            next_frontier.append(neighbor)
            frontier = next_frontier

        return scores

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def checkpoint(self, name: str, domains: list[str] | None = None) -> str:
        checkpoint_id = str(uuid.uuid4())
        if domains:
            # Partial checkpoint
            nodes = {nid: copy.deepcopy(n) for nid, n in self._nodes.items() if n.domain in domains}
            node_ids = set(nodes.keys())
            edges = [copy.deepcopy(e) for e in self._edges if e.source_id in node_ids or e.target_id in node_ids]
            rules = {rid: copy.deepcopy(r) for rid, r in self._rules.items() if r.domain in domains}
            domain_objs = {d: copy.deepcopy(self._domains[d]) for d in domains if d in self._domains}
        else:
            nodes = copy.deepcopy(self._nodes)
            edges = copy.deepcopy(self._edges)
            rules = copy.deepcopy(self._rules)
            domain_objs = copy.deepcopy(self._domains)

        self._checkpoints[checkpoint_id] = {
            "name": name,
            "domains": domain_objs,
            "nodes": nodes,
            "edges": edges,
            "rules": rules,
            "created_at": datetime.now(timezone.utc),
        }
        return checkpoint_id

    def restore(self, checkpoint_id: str) -> None:
        if checkpoint_id not in self._checkpoints:
            raise ValueError(f"Checkpoint {checkpoint_id} not found")
        cp = self._checkpoints[checkpoint_id]
        self._domains = copy.deepcopy(cp["domains"])
        self._nodes = copy.deepcopy(cp["nodes"])
        self._edges = copy.deepcopy(cp["edges"])
        self._rules = copy.deepcopy(cp["rules"])

    def list_checkpoints(self) -> list[dict]:
        return [
            {
                "id": cid,
                "name": cp["name"],
                "created_at": cp["created_at"],
            }
            for cid, cp in self._checkpoints.items()
        ]

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _recount_domain(self, domain_name: str) -> None:
        """Recount stats for a domain."""
        if domain_name not in self._domains:
            return
        domain = self._domains[domain_name]
        domain.concept_count = sum(1 for n in self._nodes.values() if n.domain == domain_name)
        domain_node_ids = {nid for nid, n in self._nodes.items() if n.domain == domain_name}
        domain.edge_count = sum(
            1 for e in self._edges
            if e.source_id in domain_node_ids or e.target_id in domain_node_ids
        )
        domain.rule_count = sum(1 for r in self._rules.values() if r.domain == domain_name)
