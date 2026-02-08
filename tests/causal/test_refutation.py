"""Tests for DAGRefuter.

Skips if scipy is not installed.
"""

import pytest

from qortex.causal.types import IndependenceAssertion

try:
    import scipy  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

pytestmark = pytest.mark.skipif(not HAS_SCIPY, reason="scipy not installed")


@pytest.fixture
def refuter():
    from qortex.causal.refutation import DAGRefuter

    return DAGRefuter(significance_level=0.05)


@pytest.fixture
def independent_data():
    """Two columns that are independent (random)."""
    import random

    random.seed(42)
    n = 200
    return {
        "X": [random.choice(["a", "b"]) for _ in range(n)],
        "Y": [random.choice(["c", "d"]) for _ in range(n)],
    }


@pytest.fixture
def dependent_data():
    """Two columns that are strongly dependent (Y copies X)."""
    n = 200
    x = ["a", "b"] * (n // 2)
    y = x.copy()  # Perfect dependence
    return {"X": x, "Y": y}


class TestMarginalIndependence:
    def test_independent_data_consistent(self, refuter, independent_data):
        assertion = IndependenceAssertion(
            x=frozenset({"X"}),
            y=frozenset({"Y"}),
            z=frozenset(),
            is_independent=True,
            method="d_separation",
        )
        result = refuter.test_independence(assertion, independent_data)
        assert result.consistent == True  # noqa: E712 — numpy bool fails `is True`
        assert result.p_value > 0.05

    def test_dependent_data_consistent(self, refuter, dependent_data):
        assertion = IndependenceAssertion(
            x=frozenset({"X"}),
            y=frozenset({"Y"}),
            z=frozenset(),
            is_independent=False,
            method="d_separation",
        )
        result = refuter.test_independence(assertion, dependent_data)
        assert result.consistent == True  # noqa: E712 — numpy bool fails `is True`
        assert result.p_value < 0.05

    def test_wrong_claim_detected(self, refuter, dependent_data):
        """Claim independence but data says dependent — inconsistent."""
        assertion = IndependenceAssertion(
            x=frozenset({"X"}),
            y=frozenset({"Y"}),
            z=frozenset(),
            is_independent=True,
            method="d_separation",
        )
        result = refuter.test_independence(assertion, dependent_data)
        assert result.consistent == False  # noqa: E712 — numpy bool fails `is True`


class TestConditionalIndependence:
    def test_conditional_test(self, refuter):
        """Simple conditional independence test."""
        import random

        random.seed(123)
        n = 300
        z = [random.choice(["z0", "z1"]) for _ in range(n)]
        x = [random.choice(["a", "b"]) for _ in range(n)]
        y = [random.choice(["c", "d"]) for _ in range(n)]
        data = {"X": x, "Y": y, "Z": z}

        assertion = IndependenceAssertion(
            x=frozenset({"X"}),
            y=frozenset({"Y"}),
            z=frozenset({"Z"}),
            is_independent=True,
            method="d_separation",
        )
        result = refuter.test_independence(assertion, data)
        assert result.sample_size == 300


class TestEdgeCases:
    def test_missing_data_raises(self, refuter):
        assertion = IndependenceAssertion(
            x=frozenset({"X"}),
            y=frozenset({"MISSING"}),
            z=frozenset(),
            is_independent=True,
            method="d_separation",
        )
        with pytest.raises(ValueError, match="Missing data"):
            refuter.test_independence(assertion, {"X": [1, 2, 3]})

    def test_insufficient_data_raises(self, refuter):
        assertion = IndependenceAssertion(
            x=frozenset({"X"}),
            y=frozenset({"Y"}),
            z=frozenset(),
            is_independent=True,
            method="d_separation",
        )
        with pytest.raises(ValueError, match="Insufficient data"):
            refuter.test_independence(assertion, {"X": [1, 2], "Y": [3, 4]})


class TestRefuteAll:
    def test_skips_bad_assertions(self, refuter, independent_data):
        good = IndependenceAssertion(
            x=frozenset({"X"}),
            y=frozenset({"Y"}),
            z=frozenset(),
            is_independent=True,
            method="d_separation",
        )
        bad = IndependenceAssertion(
            x=frozenset({"X"}),
            y=frozenset({"MISSING"}),
            z=frozenset(),
            is_independent=True,
            method="d_separation",
        )
        results = refuter.refute_all([good, bad], independent_data)
        assert len(results) == 1  # Only the good one processed
