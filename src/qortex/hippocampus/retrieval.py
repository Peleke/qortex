"""HippoRAG-style retrieval with graceful degradation."""

from __future__ import annotations

from dataclasses import dataclass

from qortex.core.backend import GraphBackend
from qortex.core.models import Rule


@dataclass
class RetrievalResult:
    """Result of a hippocampus query."""

    rules: list[Rule]
    activated_concepts: dict[str, list[str]]  # domain -> concept names
    scores: dict[str, float]  # concept_id -> relevance score


class Hippocampus:
    """Cross-domain retrieval via HippoRAG pattern completion.

    Query flow:
    1. Extract concepts from query (NER or keyword)
    2. Find matching concepts in graph (sparse index)
    3. Pattern completion: expand via PPR or BFS
    4. Collect rules from activated concepts
    5. Rank and return

    Graceful degradation:
    - Full: PPR via MAGE for pattern completion
    - Fallback: BFS traversal (less accurate but works)
    """

    def __init__(self, backend: GraphBackend):
        self.backend = backend
        self._use_ppr = backend.supports_mage()

    def query(
        self,
        context: str,
        domains: list[str] | None = None,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Retrieve relevant rules for a context.

        Args:
            context: Query context (e.g., file content, task description)
            domains: Limit to these domains (None = all)
            top_k: Number of rules to return

        Returns:
            RetrievalResult with rules, activated concepts, and scores
        """
        # 1. Extract concepts from query
        query_concepts = self._extract_query_concepts(context)

        if not query_concepts:
            return RetrievalResult(rules=[], activated_concepts={}, scores={})

        # 2. Find matching nodes in graph
        seed_nodes = self._find_seed_nodes(query_concepts, domains)

        if not seed_nodes:
            return RetrievalResult(rules=[], activated_concepts={}, scores={})

        # 3. Pattern completion
        if self._use_ppr:
            scores = self._ppr_completion(seed_nodes, domains)
        else:
            scores = self._bfs_completion(seed_nodes, domains)

        # 4. Collect activated concepts by domain
        activated = self._group_by_domain(scores, threshold=0.01)

        # 5. Retrieve rules from activated concepts
        rules = self._collect_rules(scores, top_k)

        return RetrievalResult(
            rules=rules,
            activated_concepts=activated,
            scores=scores,
        )

    def _extract_query_concepts(self, context: str) -> list[str]:
        """Extract concept keywords from query.

        TODO M3: Use NER model for proper extraction.
        For now: simple keyword extraction.
        """
        # Placeholder: split on whitespace, filter short words
        words = context.lower().split()
        return [w for w in words if len(w) > 3]

    def _find_seed_nodes(
        self,
        query_concepts: list[str],
        domains: list[str] | None,
    ) -> list[str]:
        """Find graph nodes matching query concepts."""
        seed_ids = []

        for concept in query_concepts:
            # Search for nodes with matching names
            nodes = self.backend.find_nodes(
                domain=None,  # Search all domains, filter later
                name_pattern=concept,
                limit=5,
            )
            for node in nodes:
                if domains is None or node.domain in domains:
                    seed_ids.append(node.id)

        return list(set(seed_ids))

    def _ppr_completion(
        self,
        seed_nodes: list[str],
        domains: list[str] | None,
    ) -> dict[str, float]:
        """Pattern completion via Personalized PageRank.

        This is the full HippoRAG approach.
        """
        # PPR from seed nodes
        scores = self.backend.personalized_pagerank(
            source_nodes=seed_nodes,
            damping_factor=0.85,
            max_iterations=100,
            domain=None,  # Cross-domain
        )

        # Filter by domains if specified
        if domains:
            scores = {
                nid: score for nid, score in scores.items() if self._node_in_domains(nid, domains)
            }

        return scores

    def _bfs_completion(
        self,
        seed_nodes: list[str],
        domains: list[str] | None,
        max_depth: int = 2,
    ) -> dict[str, float]:
        """Fallback: BFS traversal from seed nodes.

        Less accurate than PPR but works without MAGE.
        """
        scores: dict[str, float] = {}
        visited: set[str] = set()
        frontier = [(nid, 1.0) for nid in seed_nodes]

        for depth in range(max_depth + 1):
            next_frontier = []
            decay = 0.5**depth

            for node_id, base_score in frontier:
                if node_id in visited:
                    continue
                visited.add(node_id)

                # Check domain filter
                if domains and not self._node_in_domains(node_id, domains):
                    continue

                score = base_score * decay
                scores[node_id] = max(scores.get(node_id, 0), score)

                # Expand to neighbors
                for edge in self.backend.get_edges(node_id, direction="both"):
                    neighbor = edge.target_id if edge.source_id == node_id else edge.source_id
                    if neighbor not in visited:
                        next_frontier.append((neighbor, score))

            frontier = next_frontier

        return scores

    def _node_in_domains(self, node_id: str, domains: list[str]) -> bool:
        """Check if a node belongs to allowed domains."""
        # Node IDs are formatted as "domain:name"
        domain = node_id.split(":")[0] if ":" in node_id else None
        return domain in domains if domain else False

    def _group_by_domain(
        self,
        scores: dict[str, float],
        threshold: float,
    ) -> dict[str, list[str]]:
        """Group activated concepts by domain."""
        by_domain: dict[str, list[str]] = {}

        for node_id, score in scores.items():
            if score < threshold:
                continue

            parts = node_id.split(":", 1)
            if len(parts) == 2:
                domain, name = parts
                if domain not in by_domain:
                    by_domain[domain] = []
                by_domain[domain].append(name)

        return by_domain

    def _collect_rules(
        self,
        scores: dict[str, float],
        top_k: int,
    ) -> list[Rule]:
        """Collect rules from activated concepts.

        TODO M4: Implement rule collection from graph.
        """
        # Placeholder: return empty list
        # Real implementation would:
        # 1. Get rules linked to activated concepts
        # 2. Derive rules from edges (Phase B)
        # 3. Score by concept activation
        # 4. Return top-k
        return []
