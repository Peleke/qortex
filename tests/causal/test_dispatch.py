"""Tests for CausalDispatcher."""

import pytest

from qortex.causal.backend import NetworkXCausalBackend
from qortex.causal.dispatch import CausalDispatcher
from qortex.causal.types import CausalCapability, CausalQuery, QueryType


class TestAutoDetect:
    def test_detects_networkx(self, chain_dag):
        dispatcher = CausalDispatcher.auto_detect(chain_dag)
        assert dispatcher.backend_name == "networkx"
        assert isinstance(dispatcher.backend, NetworkXCausalBackend)

    def test_has_d_separation_capability(self, chain_dag):
        dispatcher = CausalDispatcher.auto_detect(chain_dag)
        assert dispatcher.has_capability(CausalCapability.D_SEPARATION)

    def test_no_intervention_capability(self, chain_dag):
        dispatcher = CausalDispatcher.auto_detect(chain_dag)
        assert not dispatcher.has_capability(CausalCapability.INTERVENTION)


class TestQuery:
    def test_observational_query(self, chain_dag):
        dispatcher = CausalDispatcher.auto_detect(chain_dag)
        cq = CausalQuery(
            query_type=QueryType.OBSERVATIONAL,
            target_nodes=frozenset({"A"}),
            conditioning_nodes=frozenset({"C"}),
        )
        result = dispatcher.query(cq)
        assert result is not None

    def test_interventional_raises(self, chain_dag):
        dispatcher = CausalDispatcher.auto_detect(chain_dag)
        cq = CausalQuery(
            query_type=QueryType.INTERVENTIONAL,
            target_nodes=frozenset({"A"}),
        )
        with pytest.raises(NotImplementedError):
            dispatcher.query(cq)


class TestNoBackend:
    def test_no_backend_raises_on_query(self):
        dispatcher = CausalDispatcher()
        cq = CausalQuery(
            query_type=QueryType.OBSERVATIONAL,
            target_nodes=frozenset({"A"}),
        )
        with pytest.raises(RuntimeError, match="No causal backend"):
            dispatcher.query(cq)
