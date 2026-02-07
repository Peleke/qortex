"""Tests for CausalBackend protocol and NetworkXCausalBackend."""

import pytest

from qortex.causal.backend import CausalBackend, NetworkXCausalBackend
from qortex.causal.dag import CausalDAG
from qortex.causal.types import (
    CausalCapability,
    CausalDirection,
    CausalEdge,
    CausalQuery,
    QueryType,
)


@pytest.fixture
def nx_backend(chain_dag) -> NetworkXCausalBackend:
    return NetworkXCausalBackend(dag=chain_dag)


class TestProtocolCompliance:
    def test_is_causal_backend(self, nx_backend):
        assert isinstance(nx_backend, CausalBackend)


class TestCapabilities:
    def test_capabilities_returns_frozenset(self, nx_backend):
        caps = nx_backend.capabilities()
        assert isinstance(caps, frozenset)

    def test_has_d_separation(self, nx_backend):
        assert nx_backend.has_capability(CausalCapability.D_SEPARATION)

    def test_no_intervention(self, nx_backend):
        assert not nx_backend.has_capability(CausalCapability.INTERVENTION)

    def test_no_counterfactual(self, nx_backend):
        assert not nx_backend.has_capability(CausalCapability.COUNTERFACTUAL)

    def test_no_probabilistic_inference(self, nx_backend):
        assert not nx_backend.has_capability(CausalCapability.PROBABILISTIC_INFERENCE)


class TestDSeparation:
    def test_is_d_separated_chain(self, nx_backend):
        result = nx_backend.is_d_separated(frozenset({"A"}), frozenset({"C"}), frozenset({"B"}))
        assert result.is_independent is True
        assert result.method == "d_separation"

    def test_not_d_separated_chain(self, nx_backend):
        result = nx_backend.is_d_separated(frozenset({"A"}), frozenset({"C"}), frozenset())
        assert result.is_independent is False


class TestQuery:
    def test_observational_query(self, nx_backend):
        cq = CausalQuery(
            query_type=QueryType.OBSERVATIONAL,
            target_nodes=frozenset({"A"}),
            conditioning_nodes=frozenset({"C"}),
        )
        result = nx_backend.query(cq)
        assert result.backend_used == "networkx"
        assert len(result.independences) > 0

    def test_interventional_raises(self, nx_backend):
        cq = CausalQuery(
            query_type=QueryType.INTERVENTIONAL,
            target_nodes=frozenset({"A"}),
        )
        with pytest.raises(NotImplementedError):
            nx_backend.query(cq)

    def test_counterfactual_raises(self, nx_backend):
        cq = CausalQuery(
            query_type=QueryType.COUNTERFACTUAL,
            target_nodes=frozenset({"A"}),
        )
        with pytest.raises(NotImplementedError):
            nx_backend.query(cq)


class TestEngineProperty:
    def test_engine_lazily_created(self):
        dag = CausalDAG.from_edges(
            [CausalEdge("X", "Y", "r", CausalDirection.FORWARD)],
        )
        backend = NetworkXCausalBackend(dag=dag)
        assert backend._engine is None
        _ = backend.engine  # Access triggers creation
        assert backend._engine is not None

    def test_engine_reused(self, nx_backend):
        engine1 = nx_backend.engine
        engine2 = nx_backend.engine
        assert engine1 is engine2
