"""DSeparationEngine — structural independence via networkx.is_d_separator().

Requires networkx >= 3.3 for ``nx.is_d_separator()``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import networkx as nx

from .dag import CausalDAG
from .types import CausalQuery, CausalResult, IndependenceAssertion, QueryType


@dataclass
class DSeparationEngine:
    """Answers d-separation queries on a CausalDAG."""

    dag: CausalDAG

    # ------------------------------------------------------------------
    # Core query
    # ------------------------------------------------------------------

    def is_d_separated(
        self,
        x: frozenset[str],
        y: frozenset[str],
        z: frozenset[str],
    ) -> IndependenceAssertion:
        """Test whether *x* and *y* are d-separated given *z*."""
        result = nx.is_d_separator(self.dag.graph, x, y, z)
        return IndependenceAssertion(
            x=x,
            y=y,
            z=z,
            is_independent=result,
            method="d_separation",
        )

    def query(self, cq: CausalQuery) -> CausalResult:
        """Answer a CausalQuery (OBSERVATIONAL only at Phase 1).

        Raises NotImplementedError for INTERVENTIONAL / COUNTERFACTUAL.
        """
        if cq.query_type != QueryType.OBSERVATIONAL:
            raise NotImplementedError(
                f"{cq.query_type.value} queries require Phase 2+ backends (Pyro/ChiRho)"
            )

        independences: list[IndependenceAssertion] = []
        if cq.target_nodes and cq.conditioning_nodes:
            assertion = self.is_d_separated(
                cq.target_nodes,
                cq.conditioning_nodes,
                frozenset(),
            )
            independences.append(assertion)

        return CausalResult(
            query=cq,
            independences=independences,
            backend_used="networkx",
            capabilities_used=["d_separation"],
        )

    # ------------------------------------------------------------------
    # Exhaustive enumeration
    # ------------------------------------------------------------------

    def find_all_d_separations(
        self,
        max_conditioning_size: int = 3,
    ) -> list[IndependenceAssertion]:
        """Find all d-separation relations up to a conditioning set size bound.

        Enumerates all (x, y) pairs of individual nodes and all subsets of
        remaining nodes up to ``max_conditioning_size`` as the conditioning set.
        """
        nodes = sorted(self.dag.node_ids)
        results: list[IndependenceAssertion] = []

        for i, x_id in enumerate(nodes):
            for y_id in nodes[i + 1 :]:
                remaining = [n for n in nodes if n != x_id and n != y_id]

                for size in range(0, min(max_conditioning_size + 1, len(remaining) + 1)):
                    for z_tuple in itertools.combinations(remaining, size):
                        z = frozenset(z_tuple)
                        assertion = self.is_d_separated(
                            frozenset({x_id}),
                            frozenset({y_id}),
                            z,
                        )
                        if assertion.is_independent:
                            results.append(assertion)

        return results

    def find_minimal_conditioning_set(
        self,
        x: str,
        y: str,
    ) -> frozenset[str] | None:
        """Find the smallest conditioning set that d-separates x and y.

        Returns None if x and y cannot be d-separated by any subset of
        remaining nodes.
        """
        nodes = sorted(self.dag.node_ids)
        remaining = [n for n in nodes if n != x and n != y]

        # Try empty set first
        if nx.is_d_separator(self.dag.graph, {x}, {y}, set()):
            return frozenset()

        # Increase conditioning set size
        for size in range(1, len(remaining) + 1):
            for z_tuple in itertools.combinations(remaining, size):
                z = frozenset(z_tuple)
                if nx.is_d_separator(self.dag.graph, {x}, {y}, z):
                    return z

        return None
