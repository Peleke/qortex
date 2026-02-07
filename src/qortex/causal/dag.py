"""CausalDAG — acyclic directed graph for causal reasoning.

Wraps networkx.DiGraph. Import-guarded for ``pip install qortex[causal]``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .types import (
    RELATION_CAUSAL_DIRECTION,
    CausalDirection,
    CausalEdge,
    CausalNode,
)

if TYPE_CHECKING:
    from qortex.core.backend import GraphBackend

logger = logging.getLogger(__name__)

try:
    import networkx as nx
except ImportError as _nx_err:
    raise ImportError(
        "networkx is required for the causal module: pip install qortex[causal]"
    ) from _nx_err


@dataclass
class CausalDAG:
    """Acyclic directed graph for causal reasoning.

    Wraps ``networkx.DiGraph`` with causal-aware construction,
    cycle-breaking, and accessor helpers.
    """

    _graph: nx.DiGraph = field(default_factory=nx.DiGraph, repr=False)
    _nodes: dict[str, CausalNode] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_edges(
        cls,
        edges: list[CausalEdge],
        node_names: dict[str, str] | None = None,
    ) -> CausalDAG:
        """Build a CausalDAG from explicit edges (useful for testing).

        Args:
            edges: Directed causal edges.
            node_names: Optional id→name mapping. If absent, name == id.
        """
        dag = cls()
        names = node_names or {}

        # Collect node ids from edges
        node_ids: set[str] = set()
        for e in edges:
            node_ids.add(e.source_id)
            node_ids.add(e.target_id)

        for nid in node_ids:
            node = CausalNode(id=nid, name=names.get(nid, nid), domain="")
            dag._nodes[nid] = node
            dag._graph.add_node(nid, **{"name": node.name})

        for e in edges:
            dag._graph.add_edge(
                e.source_id,
                e.target_id,
                relation_type=e.relation_type,
                strength=e.strength,
            )

        dag._break_cycles()
        return dag

    @classmethod
    def from_backend(
        cls,
        backend: GraphBackend,
        domain: str,
        relation_mapping: dict[str, tuple[CausalDirection, float]] | None = None,
    ) -> CausalDAG:
        """Build a CausalDAG from a GraphBackend's nodes and edges.

        Filters out NONE and BIDIRECTIONAL directions (no DAG representation).
        Reverses edges with REVERSE direction.

        Args:
            backend: A connected GraphBackend instance.
            domain: Domain to read from.
            relation_mapping: Override for RELATION_CAUSAL_DIRECTION.
        """
        mapping = relation_mapping or RELATION_CAUSAL_DIRECTION
        dag = cls()

        # Load nodes
        for node in backend.find_nodes(domain=domain, limit=100_000):
            cn = CausalNode(
                id=node.id,
                name=node.name,
                domain=node.domain,
                properties=node.properties,
            )
            dag._nodes[node.id] = cn
            dag._graph.add_node(node.id, **{"name": node.name})

        # Load edges
        for nid in list(dag._nodes):
            for edge in backend.get_edges(nid, direction="out"):
                rel = (
                    edge.relation_type.value
                    if hasattr(edge.relation_type, "value")
                    else str(edge.relation_type)
                )
                if rel not in mapping:
                    continue

                direction, default_strength = mapping[rel]

                # Skip non-DAG edges
                if direction in (CausalDirection.BIDIRECTIONAL, CausalDirection.NONE):
                    continue

                strength = edge.confidence * default_strength

                if direction == CausalDirection.FORWARD:
                    src, tgt = edge.source_id, edge.target_id
                else:  # REVERSE
                    src, tgt = edge.target_id, edge.source_id

                # Only add if both nodes exist in this domain
                if src in dag._nodes and tgt in dag._nodes:
                    dag._graph.add_edge(
                        src,
                        tgt,
                        relation_type=rel,
                        strength=strength,
                    )

        dag._break_cycles()
        return dag

    # ------------------------------------------------------------------
    # Cycle breaking
    # ------------------------------------------------------------------

    def _break_cycles(self) -> None:
        """Remove lowest-strength edges until the graph is acyclic."""
        while True:
            try:
                cycle = nx.find_cycle(self._graph)
            except nx.NetworkXNoCycle:
                break

            # Find weakest edge in the cycle
            weakest_edge: tuple[str, str] | None = None
            weakest_strength = float("inf")
            for u, v, *_ in cycle:
                s = self._graph.edges[u, v].get("strength", 1.0)
                if s < weakest_strength:
                    weakest_strength = s
                    weakest_edge = (u, v)

            if weakest_edge:
                logger.info(
                    "Breaking cycle: removing edge %s→%s (strength=%.3f)",
                    weakest_edge[0],
                    weakest_edge[1],
                    weakest_strength,
                )
                self._graph.remove_edge(*weakest_edge)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        """The underlying networkx DiGraph."""
        return self._graph

    @property
    def nodes(self) -> dict[str, CausalNode]:
        """Node id → CausalNode mapping."""
        return dict(self._nodes)

    @property
    def node_ids(self) -> frozenset[str]:
        """All node ids in the DAG."""
        return frozenset(self._graph.nodes)

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def ancestors(self, node_id: str) -> frozenset[str]:
        """All ancestors of a node (transitive parents)."""
        return frozenset(nx.ancestors(self._graph, node_id))

    def descendants(self, node_id: str) -> frozenset[str]:
        """All descendants of a node (transitive children)."""
        return frozenset(nx.descendants(self._graph, node_id))

    def parents(self, node_id: str) -> frozenset[str]:
        """Direct parents of a node."""
        return frozenset(self._graph.predecessors(node_id))

    def children(self, node_id: str) -> frozenset[str]:
        """Direct children of a node."""
        return frozenset(self._graph.successors(node_id))

    def topological_order(self) -> list[str]:
        """Topological sort of all nodes."""
        return list(nx.topological_sort(self._graph))

    def is_valid_dag(self) -> bool:
        """Check that the graph is a valid DAG (directed, acyclic)."""
        return self._graph.is_directed() and nx.is_directed_acyclic_graph(self._graph)

    def edge_strength(self, source: str, target: str) -> float:
        """Get the strength of an edge, or 0.0 if absent."""
        data = self._graph.edges.get((source, target))
        if data is None:
            return 0.0
        return data.get("strength", 1.0)
