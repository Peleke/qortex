"""In-memory graph backend for testing without Docker/Memgraph."""

from __future__ import annotations

import copy
import fnmatch
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Literal

from qortex_observe import emit
from qortex_observe.events import ManifestIngested, PPRConverged, PPRDiverged, PPRStarted

from .backend import GraphPattern
from .models import ConceptEdge, ConceptNode, Domain, ExplicitRule, IngestionManifest


class InMemoryBackend:
    """Full GraphBackend implementation using Python dicts.

    Designed for testing — no external dependencies required.
    Supports all GraphBackend operations except Cypher queries.
    Optionally supports vector search when a VectorIndex is provided.
    """

    def __init__(self, vector_index=None) -> None:
        self._connected = False
        self._domains: dict[str, Domain] = {}
        self._nodes: dict[str, ConceptNode] = {}
        self._edges: list[ConceptEdge] = []
        self._rules: dict[str, ExplicitRule] = {}
        self._checkpoints: dict[str, dict] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._vector_index = vector_index  # Optional VectorIndex instance

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
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
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
        node_ids_to_remove = {nid for nid, node in self._nodes.items() if node.domain == name}
        for nid in node_ids_to_remove:
            del self._nodes[nid]
        # Cascade: remove edges referencing removed nodes
        self._edges = [
            e
            for e in self._edges
            if e.source_id not in node_ids_to_remove and e.target_id not in node_ids_to_remove
        ]
        # Cascade: remove rules in this domain
        rule_ids_to_remove = {rid for rid, rule in self._rules.items() if rule.domain == name}
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
            if (
                direction == "out"
                and edge.source_id == node_id
                or direction == "in"
                and edge.target_id == node_id
                or direction == "both"
                and (edge.source_id == node_id or edge.target_id == node_id)
            ):
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
        t0 = time.perf_counter()

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
        domain.updated_at = datetime.now(UTC)

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
        seed_weights: dict[str, float] | None = None,
        extra_edges: list[tuple[str, str, float]] | None = None,
        query_id: str | None = None,
    ) -> dict[str, float]:
        """Personalized PageRank via power iteration.

        Real PPR over the in-memory graph — not a BFS approximation.
        Accepts optional seed_weights (from teleportation factors) and
        extra_edges (from online edge generation) for the hybrid pipeline.

        Power iteration formula:
            π(t+1) = d * (A @ π(t)) + (1 - d) * personalization

        Args:
            source_nodes: Seed node IDs for personalization.
            damping_factor: Probability of following an edge vs teleporting (0-1).
            max_iterations: Max iterations before stopping.
            domain: Restrict to nodes in this domain (None = all).
            seed_weights: Optional per-seed teleportation weights (from factors).
                If None, all seeds get equal weight.
            extra_edges: Optional ephemeral edges [(src, tgt, weight), ...] from
                online edge generation. Merged with persistent edges for PPR.
        """
        # 1. Collect all node IDs in scope
        if domain:
            node_ids = [nid for nid, n in self._nodes.items() if n.domain == domain]
        else:
            node_ids = list(self._nodes.keys())

        # Filter source nodes to those in scope
        valid_seeds = [nid for nid in source_nodes if nid in self._nodes]
        if domain:
            valid_seeds = [nid for nid in valid_seeds if self._nodes[nid].domain == domain]

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

        # 2. Build adjacency: node_id → [(neighbor_id, weight)]
        adjacency: dict[str, list[tuple[str, float]]] = {nid: [] for nid in node_ids}

        # Persistent edges
        node_set = set(node_ids)
        for edge in self._edges:
            if edge.source_id in node_set and edge.target_id in node_set:
                adjacency[edge.source_id].append((edge.target_id, edge.confidence))
                # Undirected: add reverse edge too
                adjacency[edge.target_id].append((edge.source_id, edge.confidence))

        # Extra edges (from online edge gen)
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

        # Normalize personalization
        p_total = sum(personalization.values())
        if p_total > 0:
            personalization = {k: v / p_total for k, v in personalization.items()}

        # 4. Power iteration
        # Initialize π uniformly
        n = len(node_ids)
        scores: dict[str, float] = {nid: 1.0 / n for nid in node_ids}

        convergence_threshold = 1e-6

        for _iteration in range(max_iterations):
            new_scores: dict[str, float] = {}

            for nid in node_ids:
                # Teleportation component
                teleport = (1.0 - damping_factor) * personalization.get(nid, 0.0)

                # Walk component: sum of (neighbor_score * edge_weight / neighbor_out_degree)
                walk = 0.0
                for neighbor_id, weight in adjacency.get(nid, []):
                    neighbor_out = adjacency.get(neighbor_id, [])
                    out_weight = sum(w for _, w in neighbor_out)
                    if out_weight > 0:
                        walk += scores.get(neighbor_id, 0.0) * weight / out_weight

                new_scores[nid] = teleport + damping_factor * walk

            # Check convergence (L1 norm)
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

        # Filter out near-zero scores
        return result

    # -------------------------------------------------------------------------
    # Vector Operations
    # -------------------------------------------------------------------------

    def add_embedding(self, node_id: str, embedding: list[float]) -> None:
        """Store embedding and optionally index it."""
        self._embeddings[node_id] = embedding
        if self._vector_index is not None:
            self._vector_index.add([node_id], [embedding])

    def get_embedding(self, node_id: str) -> list[float] | None:
        return self._embeddings.get(node_id)

    def vector_search(
        self,
        query_embedding: list[float],
        domain: str | None = None,
        top_k: int = 10,
        threshold: float = 0.0,
    ) -> list[tuple[ConceptNode, float]]:
        """Vector similarity search over concept embeddings."""
        if self._vector_index is not None:
            # Use the index for fast search
            results = self._vector_index.search(
                query_embedding, top_k=top_k * 2, threshold=threshold
            )
            filtered = []
            for node_id, score in results:
                node = self._nodes.get(node_id)
                if node is None:
                    continue
                if domain and node.domain != domain:
                    continue
                filtered.append((node, score))
                if len(filtered) >= top_k:
                    break
            return filtered

        # Fallback: brute-force over stored embeddings
        if not self._embeddings:
            return []

        try:
            import numpy as np
        except ImportError:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm == 0:
            return []
        query = query / norm

        scores: list[tuple[str, float]] = []
        for node_id, emb in self._embeddings.items():
            if domain:
                node = self._nodes.get(node_id)
                if node is None or node.domain != domain:
                    continue
            vec = np.array(emb, dtype=np.float32)
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            sim = float(np.dot(query, vec / vec_norm))
            if sim >= threshold:
                scores.append((node_id, sim))

        scores.sort(key=lambda x: -x[1])
        results = []
        for node_id, score in scores[:top_k]:
            node = self._nodes.get(node_id)
            if node:
                results.append((node, score))
        return results

    def supports_vector_search(self) -> bool:
        return bool(self._embeddings) or self._vector_index is not None

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def checkpoint(self, name: str, domains: list[str] | None = None) -> str:
        checkpoint_id = str(uuid.uuid4())
        if domains:
            # Partial checkpoint
            nodes = {nid: copy.deepcopy(n) for nid, n in self._nodes.items() if n.domain in domains}
            node_ids = set(nodes.keys())
            edges = [
                copy.deepcopy(e)
                for e in self._edges
                if e.source_id in node_ids or e.target_id in node_ids
            ]
            rules = {rid: copy.deepcopy(r) for rid, r in self._rules.items() if r.domain in domains}
            domain_objs = {
                d: copy.deepcopy(self._domains[d]) for d in domains if d in self._domains
            }
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
            "created_at": datetime.now(UTC),
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
            1
            for e in self._edges
            if e.source_id in domain_node_ids or e.target_id in domain_node_ids
        )
        domain.rule_count = sum(1 for r in self._rules.values() if r.domain == domain_name)
