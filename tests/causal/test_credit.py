"""Tests for CreditAssigner."""

import pytest

from qortex.causal.credit import CreditAssigner
from qortex.causal.dag import CausalDAG
from qortex.causal.types import CausalDirection, CausalEdge


@pytest.fixture
def credit_dag() -> CausalDAG:
    """A → B → C → D (chain for credit propagation)."""
    return CausalDAG.from_edges(
        [
            CausalEdge("A", "B", "r", CausalDirection.FORWARD, strength=0.9),
            CausalEdge("B", "C", "r", CausalDirection.FORWARD, strength=0.8),
            CausalEdge("C", "D", "r", CausalDirection.FORWARD, strength=0.7),
        ],
        node_names={"A": "A", "B": "B", "C": "C", "D": "D"},
    )


class TestAssignCredit:
    def test_direct_credit(self, credit_dag):
        assigner = CreditAssigner(dag=credit_dag)
        assignments = assigner.assign_credit(["D"], reward=1.0)
        direct = [a for a in assignments if a.method == "direct"]
        assert len(direct) == 1
        assert direct[0].concept_id == "D"
        assert direct[0].credit == 1.0

    def test_ancestor_credit_decays(self, credit_dag):
        assigner = CreditAssigner(dag=credit_dag, decay_factor=0.5)
        assignments = assigner.assign_credit(["D"], reward=1.0)
        ancestor = {a.concept_id: a for a in assignments if a.method == "ancestor"}

        # C is parent of D: 1.0 * 0.5 * 0.7 = 0.35
        assert "C" in ancestor
        assert abs(ancestor["C"].credit - 0.35) < 0.01

        # B is grandparent: 0.35 * 0.5 * 0.8 = 0.14
        assert "B" in ancestor
        assert abs(ancestor["B"].credit - 0.14) < 0.01

    def test_threshold_filters_small_credit(self, credit_dag):
        assigner = CreditAssigner(dag=credit_dag, decay_factor=0.1, min_credit=0.05)
        assignments = assigner.assign_credit(["D"], reward=1.0)
        # With rapid decay, distant ancestors should be filtered
        for a in assignments:
            assert abs(a.credit) >= 0.05 or a.method == "direct"

    def test_negative_reward(self, credit_dag):
        assigner = CreditAssigner(dag=credit_dag)
        assignments = assigner.assign_credit(["D"], reward=-1.0)
        direct = [a for a in assignments if a.method == "direct"]
        assert direct[0].credit == -1.0

    def test_nonexistent_concept_ignored(self, credit_dag):
        assigner = CreditAssigner(dag=credit_dag)
        assignments = assigner.assign_credit(["NONEXISTENT"], reward=1.0)
        assert assignments == []

    def test_multiple_rule_concepts(self, credit_dag):
        assigner = CreditAssigner(dag=credit_dag)
        assignments = assigner.assign_credit(["C", "D"], reward=1.0)
        direct = [a for a in assignments if a.method == "direct"]
        assert len(direct) == 2

    def test_paths_recorded(self, credit_dag):
        assigner = CreditAssigner(dag=credit_dag, decay_factor=0.5)
        assignments = assigner.assign_credit(["D"], reward=1.0)
        for a in assignments:
            assert len(a.path) >= 1
            # Direct assignments have path [concept_id]
            if a.method == "direct":
                assert a.path == [a.concept_id]


class TestPosteriorUpdates:
    def test_positive_credit_updates_alpha(self, credit_dag):
        assigner = CreditAssigner(dag=credit_dag)
        assignments = assigner.assign_credit(["D"], reward=1.0)
        updates = CreditAssigner.to_posterior_updates(assignments)

        assert "D" in updates
        assert updates["D"]["alpha_delta"] > 0
        assert updates["D"]["beta_delta"] == 0.0

    def test_negative_credit_updates_beta(self, credit_dag):
        assigner = CreditAssigner(dag=credit_dag)
        assignments = assigner.assign_credit(["D"], reward=-1.0)
        updates = CreditAssigner.to_posterior_updates(assignments)

        assert "D" in updates
        assert updates["D"]["alpha_delta"] == 0.0
        assert updates["D"]["beta_delta"] > 0

    def test_all_assigned_concepts_in_updates(self, credit_dag):
        assigner = CreditAssigner(dag=credit_dag, decay_factor=0.5)
        assignments = assigner.assign_credit(["D"], reward=1.0)
        updates = CreditAssigner.to_posterior_updates(assignments)

        for a in assignments:
            assert a.concept_id in updates
