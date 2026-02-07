"""Tests for CausalDAG."""

from hypothesis import given, settings
from hypothesis import strategies as st

from qortex.causal.dag import CausalDAG
from qortex.causal.types import CausalDirection, CausalEdge


class TestFromEdges:
    def test_chain(self, chain_dag):
        assert chain_dag.is_valid_dag()
        assert chain_dag.node_ids == frozenset({"A", "B", "C"})
        assert chain_dag.edge_count == 2

    def test_fork(self, fork_dag):
        assert fork_dag.is_valid_dag()
        assert fork_dag.edge_count == 2

    def test_collider(self, collider_dag):
        assert collider_dag.is_valid_dag()
        assert collider_dag.edge_count == 2

    def test_empty(self):
        dag = CausalDAG.from_edges([], node_names={})
        assert dag.is_valid_dag()
        assert dag.node_ids == frozenset()
        assert dag.edge_count == 0


class TestCycleBreaking:
    def test_simple_cycle_removed(self):
        """A→B→C→A should have weakest edge removed."""
        edges = [
            CausalEdge("A", "B", "r", CausalDirection.FORWARD, strength=0.9),
            CausalEdge("B", "C", "r", CausalDirection.FORWARD, strength=0.8),
            CausalEdge("C", "A", "r", CausalDirection.FORWARD, strength=0.3),
        ]
        dag = CausalDAG.from_edges(edges)
        assert dag.is_valid_dag()
        # Weakest edge (C→A, 0.3) should have been removed
        assert dag.edge_count == 2

    def test_preserves_strongest_edges(self):
        edges = [
            CausalEdge("X", "Y", "r", CausalDirection.FORWARD, strength=1.0),
            CausalEdge("Y", "Z", "r", CausalDirection.FORWARD, strength=1.0),
            CausalEdge("Z", "X", "r", CausalDirection.FORWARD, strength=0.1),
        ]
        dag = CausalDAG.from_edges(edges)
        assert dag.is_valid_dag()
        # X→Y and Y→Z kept, Z→X removed
        assert dag.edge_strength("X", "Y") == 1.0
        assert dag.edge_strength("Y", "Z") == 1.0
        assert dag.edge_strength("Z", "X") == 0.0


class TestAccessors:
    def test_ancestors(self, chain_dag):
        assert chain_dag.ancestors("C") == frozenset({"A", "B"})
        assert chain_dag.ancestors("A") == frozenset()

    def test_descendants(self, chain_dag):
        assert chain_dag.descendants("A") == frozenset({"B", "C"})
        assert chain_dag.descendants("C") == frozenset()

    def test_parents(self, chain_dag):
        assert chain_dag.parents("B") == frozenset({"A"})
        assert chain_dag.parents("A") == frozenset()

    def test_children(self, chain_dag):
        assert chain_dag.children("A") == frozenset({"B"})
        assert chain_dag.children("C") == frozenset()

    def test_topological_order(self, chain_dag):
        order = chain_dag.topological_order()
        assert order.index("A") < order.index("B") < order.index("C")

    def test_nodes_dict(self, chain_dag):
        nodes = chain_dag.nodes
        assert set(nodes.keys()) == {"A", "B", "C"}
        assert nodes["A"].name == "A"

    def test_edge_strength(self, chain_dag):
        assert chain_dag.edge_strength("A", "B") == 1.0
        assert chain_dag.edge_strength("A", "C") == 0.0  # No direct edge


class TestFromBackend:
    def test_builds_dag_from_backend(self, backend_with_graph):
        dag = CausalDAG.from_backend(backend_with_graph, domain="test")
        assert dag.is_valid_dag()
        assert dag.node_ids == frozenset({"A", "B", "C", "D"})
        assert dag.edge_count >= 2  # At least requires + uses edges

    def test_filters_non_dag_relations(self, backend_with_graph):
        """SIMILAR_TO and CONTRADICTS should be excluded."""
        from qortex.core.models import ConceptEdge as CE
        from qortex.core.models import RelationType

        # Add a similar_to edge — should be excluded
        backend_with_graph.add_edge(
            CE(source_id="A", target_id="D", relation_type=RelationType.SIMILAR_TO)
        )
        dag = CausalDAG.from_backend(backend_with_graph, domain="test")
        # Should still be 3 edges (A→B, B→C, C→D), not 4
        assert dag.edge_count == 3


class TestPropertyBased:
    @given(
        edges=st.lists(
            st.tuples(
                st.sampled_from(["n0", "n1", "n2", "n3", "n4"]),
                st.sampled_from(["n0", "n1", "n2", "n3", "n4"]),
            ).filter(lambda t: t[0] != t[1]),
            min_size=0,
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_always_produces_valid_dag(self, edges):
        """Any random edge list should produce a valid DAG after cycle breaking."""
        causal_edges = [
            CausalEdge(
                source_id=src,
                target_id=tgt,
                relation_type="requires",
                direction=CausalDirection.FORWARD,
                strength=0.5,
            )
            for src, tgt in edges
        ]
        dag = CausalDAG.from_edges(causal_edges)
        assert dag.is_valid_dag()
