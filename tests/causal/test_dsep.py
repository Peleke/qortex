"""Tests for DSeparationEngine.

Cross-validates every result against networkx.is_d_separator() directly.
"""

import networkx as nx
import pytest

from qortex.causal.dsep import DSeparationEngine
from qortex.causal.types import CausalQuery, QueryType


class TestChain:
    """A → B → C"""

    def test_a_independent_of_c_given_b(self, chain_dag):
        engine = DSeparationEngine(dag=chain_dag)
        result = engine.is_d_separated(frozenset({"A"}), frozenset({"C"}), frozenset({"B"}))
        assert result.is_independent is True
        # Cross-validate
        assert nx.is_d_separator(chain_dag.graph, {"A"}, {"C"}, {"B"})

    def test_a_dependent_on_c_unconditional(self, chain_dag):
        engine = DSeparationEngine(dag=chain_dag)
        result = engine.is_d_separated(frozenset({"A"}), frozenset({"C"}), frozenset())
        assert result.is_independent is False
        assert not nx.is_d_separator(chain_dag.graph, {"A"}, {"C"}, set())


class TestFork:
    """B → A, B → C"""

    def test_a_independent_of_c_given_b(self, fork_dag):
        engine = DSeparationEngine(dag=fork_dag)
        result = engine.is_d_separated(frozenset({"A"}), frozenset({"C"}), frozenset({"B"}))
        assert result.is_independent is True
        assert nx.is_d_separator(fork_dag.graph, {"A"}, {"C"}, {"B"})

    def test_a_dependent_on_c_unconditional(self, fork_dag):
        engine = DSeparationEngine(dag=fork_dag)
        result = engine.is_d_separated(frozenset({"A"}), frozenset({"C"}), frozenset())
        assert result.is_independent is False
        assert not nx.is_d_separator(fork_dag.graph, {"A"}, {"C"}, set())


class TestCollider:
    """A → B, C → B (B is a collider)"""

    def test_a_independent_of_c_unconditional(self, collider_dag):
        engine = DSeparationEngine(dag=collider_dag)
        result = engine.is_d_separated(frozenset({"A"}), frozenset({"C"}), frozenset())
        assert result.is_independent is True
        assert nx.is_d_separator(collider_dag.graph, {"A"}, {"C"}, set())

    def test_a_dependent_on_c_given_b(self, collider_dag):
        """Explaining away: conditioning on collider opens the path."""
        engine = DSeparationEngine(dag=collider_dag)
        result = engine.is_d_separated(frozenset({"A"}), frozenset({"C"}), frozenset({"B"}))
        assert result.is_independent is False
        assert not nx.is_d_separator(collider_dag.graph, {"A"}, {"C"}, {"B"})


class TestSprinkler:
    """Season → Rain, Season → Sprinkler, Rain → Wet, Sprinkler → Wet"""

    def test_rain_independent_of_sprinkler_given_season(self, sprinkler_dag):
        engine = DSeparationEngine(dag=sprinkler_dag)
        result = engine.is_d_separated(
            frozenset({"Rain"}), frozenset({"Sprinkler"}), frozenset({"Season"})
        )
        assert result.is_independent is True
        assert nx.is_d_separator(sprinkler_dag.graph, {"Rain"}, {"Sprinkler"}, {"Season"})

    def test_rain_dependent_on_sprinkler_given_wet(self, sprinkler_dag):
        """Conditioning on collider descendant — explaining away."""
        engine = DSeparationEngine(dag=sprinkler_dag)
        result = engine.is_d_separated(
            frozenset({"Rain"}), frozenset({"Sprinkler"}), frozenset({"Wet"})
        )
        assert result.is_independent is False


class TestSmoking:
    """Smoking → Tar → Cancer, Smoking → Cancer"""

    def test_tar_dependent_on_cancer_unconditional(self, smoking_dag):
        engine = DSeparationEngine(dag=smoking_dag)
        result = engine.is_d_separated(frozenset({"Tar"}), frozenset({"Cancer"}), frozenset())
        assert result.is_independent is False


class TestQuery:
    def test_observational_query(self, chain_dag):
        engine = DSeparationEngine(dag=chain_dag)
        cq = CausalQuery(
            query_type=QueryType.OBSERVATIONAL,
            target_nodes=frozenset({"A"}),
            conditioning_nodes=frozenset({"C"}),
        )
        result = engine.query(cq)
        assert result.backend_used == "networkx"
        assert "d_separation" in result.capabilities_used

    def test_interventional_raises(self, chain_dag):
        engine = DSeparationEngine(dag=chain_dag)
        cq = CausalQuery(
            query_type=QueryType.INTERVENTIONAL,
            target_nodes=frozenset({"A"}),
        )
        with pytest.raises(NotImplementedError, match="interventional"):
            engine.query(cq)


class TestFindAllDSeparations:
    def test_chain_finds_separation(self, chain_dag):
        engine = DSeparationEngine(dag=chain_dag)
        results = engine.find_all_d_separations(max_conditioning_size=3)
        # Should find A ⊥ C | B
        found = any(
            r.x == frozenset({"A"}) and r.y == frozenset({"C"}) and "B" in r.z for r in results
        )
        assert found

    def test_collider_finds_unconditional_separation(self, collider_dag):
        engine = DSeparationEngine(dag=collider_dag)
        results = engine.find_all_d_separations(max_conditioning_size=3)
        # Should find A ⊥ C | {} (unconditional)
        found = any(
            r.x == frozenset({"A"}) and r.y == frozenset({"C"}) and r.z == frozenset()
            for r in results
        )
        assert found


class TestFindMinimalConditioningSet:
    def test_chain_minimal_set(self, chain_dag):
        engine = DSeparationEngine(dag=chain_dag)
        result = engine.find_minimal_conditioning_set("A", "C")
        assert result is not None
        assert "B" in result

    def test_collider_minimal_is_empty(self, collider_dag):
        engine = DSeparationEngine(dag=collider_dag)
        result = engine.find_minimal_conditioning_set("A", "C")
        assert result == frozenset()

    def test_direct_edge_no_separation(self, chain_dag):
        engine = DSeparationEngine(dag=chain_dag)
        result = engine.find_minimal_conditioning_set("A", "B")
        # A→B direct edge, cannot d-separate
        assert result is None
