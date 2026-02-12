"""Tests for credit propagation: CreditAssigner → Learner wiring via _maybe_propagate_credit."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from qortex.flags import reset_flags
from qortex.observability import reset as obs_reset


@pytest.fixture(autouse=True)
def _clean():
    obs_reset()
    reset_flags()
    yield
    obs_reset()
    reset_flags()


class TestMaybePropagateCredit:
    """Unit tests for _maybe_propagate_credit in server.py."""

    def test_returns_none_when_flag_off(self, monkeypatch):
        from qortex.mcp.server import _maybe_propagate_credit

        monkeypatch.setenv("QORTEX_CREDIT_PROPAGATION", "off")
        reset_flags()

        result = _maybe_propagate_credit("q1", {"c:a": "accepted"})
        assert result is None

    def test_returns_none_when_no_outcomes(self, monkeypatch):
        from qortex.mcp.server import _maybe_propagate_credit

        monkeypatch.setenv("QORTEX_CREDIT_PROPAGATION", "on")
        reset_flags()

        result = _maybe_propagate_credit("q1", {})
        assert result is None

    def test_returns_none_when_backend_is_none(self, monkeypatch):
        from qortex.mcp import server
        from qortex.mcp.server import _maybe_propagate_credit

        monkeypatch.setenv("QORTEX_CREDIT_PROPAGATION", "on")
        reset_flags()
        monkeypatch.setattr(server, "_backend", None)

        result = _maybe_propagate_credit("q1", {"c:a": "accepted"})
        assert result is None

    def test_returns_none_when_networkx_unavailable(self, monkeypatch):
        from qortex.mcp.server import _maybe_propagate_credit

        monkeypatch.setenv("QORTEX_CREDIT_PROPAGATION", "on")
        reset_flags()

        # Mock backend so we get past the None check
        mock_backend = MagicMock()
        mock_backend.get_node.return_value = MagicMock(domain="test")

        import qortex.mcp.server as srv
        monkeypatch.setattr(srv, "_backend", mock_backend)

        # Simulate networkx not being available
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("qortex.causal.credit", "qortex.causal.dag"):
                raise ImportError("no networkx")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = _maybe_propagate_credit("q1", {"c:a": "accepted"})
        assert result is None

    def test_returns_none_when_no_nodes_in_backend(self, monkeypatch):
        from qortex.mcp.server import _maybe_propagate_credit

        monkeypatch.setenv("QORTEX_CREDIT_PROPAGATION", "on")
        reset_flags()

        mock_backend = MagicMock()
        mock_backend.get_node.return_value = None  # concept not found

        import qortex.mcp.server as srv
        monkeypatch.setattr(srv, "_backend", mock_backend)

        result = _maybe_propagate_credit("q1", {"unknown:x": "accepted"})
        assert result is None


class TestCreditPropagationEndToEnd:
    """Full end-to-end: backend with DAG → credit → learner update."""

    def test_credit_flows_through_dag(self, monkeypatch, tmp_path):
        from qortex.causal.dag import CausalDAG
        from qortex.causal.types import CausalDirection, CausalEdge
        from qortex.core.models import ConceptNode
        from qortex.mcp.server import _maybe_propagate_credit

        monkeypatch.setenv("QORTEX_CREDIT_PROPAGATION", "on")
        reset_flags()

        # Build a mock backend with known nodes
        nodes = {
            "test:a": ConceptNode(id="test:a", name="A", description="", domain="test", source_id="test"),
            "test:b": ConceptNode(id="test:b", name="B", description="", domain="test", source_id="test"),
            "test:c": ConceptNode(id="test:c", name="C", description="", domain="test", source_id="test"),
        }

        mock_backend = MagicMock()
        mock_backend.get_node.side_effect = lambda nid, **kw: nodes.get(nid)

        # Build a DAG: a → b → c
        edges = [
            CausalEdge("test:a", "test:b", "requires", CausalDirection.FORWARD, 0.9),
            CausalEdge("test:b", "test:c", "requires", CausalDirection.FORWARD, 0.8),
        ]
        dag = CausalDAG.from_edges(edges, {n: n for n in nodes})

        import qortex.mcp.server as srv
        monkeypatch.setattr(srv, "_backend", mock_backend)
        monkeypatch.setattr(srv, "_learning_state_dir", str(tmp_path))
        # Clear any cached learners
        monkeypatch.setattr(srv, "_learners", {})

        # Patch CausalDAG.from_backend to return our test DAG
        with patch("qortex.causal.dag.CausalDAG.from_backend", return_value=dag):
            result = _maybe_propagate_credit("q1", {"test:c": "accepted"})

        assert result is not None
        assert result["direct_count"] >= 1
        assert result["concept_count"] >= 1

        # Verify learner was updated
        credit_learner = srv._learners.get("credit")
        assert credit_learner is not None

        # "test:c" got direct credit, "test:b" is parent, "test:a" is grandparent
        state_c = credit_learner.store.get("test:c")
        assert state_c.pulls >= 1
        assert state_c.alpha > 1.0  # positive credit applied

    def test_rejected_outcome_increases_beta(self, monkeypatch, tmp_path):
        from qortex.causal.dag import CausalDAG
        from qortex.causal.types import CausalDirection, CausalEdge, CausalNode
        from qortex.core.models import ConceptNode
        from qortex.mcp.server import _maybe_propagate_credit

        monkeypatch.setenv("QORTEX_CREDIT_PROPAGATION", "on")
        reset_flags()

        nodes = {
            "test:x": ConceptNode(id="test:x", name="X", description="", domain="test", source_id="test"),
            "test:y": ConceptNode(id="test:y", name="Y", description="", domain="test", source_id="test"),
        }
        mock_backend = MagicMock()
        mock_backend.get_node.side_effect = lambda nid, **kw: nodes.get(nid)

        # Two-node DAG so from_edges registers both nodes
        dag = CausalDAG.from_edges(
            [CausalEdge("test:y", "test:x", "requires", CausalDirection.FORWARD, 1.0)],
            {"test:x": "X", "test:y": "Y"},
        )

        import qortex.mcp.server as srv
        monkeypatch.setattr(srv, "_backend", mock_backend)
        monkeypatch.setattr(srv, "_learning_state_dir", str(tmp_path))
        monkeypatch.setattr(srv, "_learners", {})

        with patch("qortex.causal.dag.CausalDAG.from_backend", return_value=dag):
            result = _maybe_propagate_credit("q1", {"test:x": "rejected"})

        assert result is not None
        credit_learner = srv._learners["credit"]
        state = credit_learner.store.get("test:x")
        # Rejected → negative reward → beta_delta > 0
        assert state.beta > 1.0

    def test_partial_outcome_gives_moderate_credit(self, monkeypatch, tmp_path):
        from qortex.causal.dag import CausalDAG
        from qortex.causal.types import CausalDirection, CausalEdge
        from qortex.core.models import ConceptNode
        from qortex.mcp.server import _maybe_propagate_credit

        monkeypatch.setenv("QORTEX_CREDIT_PROPAGATION", "on")
        reset_flags()

        nodes = {
            "test:p": ConceptNode(id="test:p", name="P", description="", domain="test", source_id="test"),
            "test:q": ConceptNode(id="test:q", name="Q", description="", domain="test", source_id="test"),
        }
        mock_backend = MagicMock()
        mock_backend.get_node.side_effect = lambda nid, **kw: nodes.get(nid)

        dag = CausalDAG.from_edges(
            [CausalEdge("test:q", "test:p", "requires", CausalDirection.FORWARD, 1.0)],
            {"test:p": "P", "test:q": "Q"},
        )

        import qortex.mcp.server as srv
        monkeypatch.setattr(srv, "_backend", mock_backend)
        monkeypatch.setattr(srv, "_learning_state_dir", str(tmp_path))
        monkeypatch.setattr(srv, "_learners", {})

        with patch("qortex.causal.dag.CausalDAG.from_backend", return_value=dag):
            result = _maybe_propagate_credit("q1", {"test:p": "partial"})

        assert result is not None
        credit_learner = srv._learners["credit"]
        state = credit_learner.store.get("test:p")
        # Partial → 0.3 reward → moderate alpha boost
        assert state.alpha > 1.0
        assert state.alpha < 2.0  # not full credit

    def test_ancestor_credit_decays(self, monkeypatch, tmp_path):
        from qortex.causal.dag import CausalDAG
        from qortex.causal.types import CausalDirection, CausalEdge
        from qortex.core.models import ConceptNode
        from qortex.mcp.server import _maybe_propagate_credit

        monkeypatch.setenv("QORTEX_CREDIT_PROPAGATION", "on")
        reset_flags()

        nodes = {
            "test:root": ConceptNode(id="test:root", name="Root", description="", domain="test", source_id="test"),
            "test:mid": ConceptNode(id="test:mid", name="Mid", description="", domain="test", source_id="test"),
            "test:leaf": ConceptNode(id="test:leaf", name="Leaf", description="", domain="test", source_id="test"),
        }
        mock_backend = MagicMock()
        mock_backend.get_node.side_effect = lambda nid, **kw: nodes.get(nid)

        # root → mid → leaf
        edges = [
            CausalEdge("test:root", "test:mid", "requires", CausalDirection.FORWARD, 1.0),
            CausalEdge("test:mid", "test:leaf", "requires", CausalDirection.FORWARD, 1.0),
        ]
        dag = CausalDAG.from_edges(edges, {n: n for n in nodes})

        import qortex.mcp.server as srv
        monkeypatch.setattr(srv, "_backend", mock_backend)
        monkeypatch.setattr(srv, "_learning_state_dir", str(tmp_path))
        monkeypatch.setattr(srv, "_learners", {})

        with patch("qortex.causal.dag.CausalDAG.from_backend", return_value=dag):
            result = _maybe_propagate_credit("q1", {"test:leaf": "accepted"})

        assert result is not None
        assert result["ancestor_count"] >= 1

        credit_learner = srv._learners["credit"]
        leaf_state = credit_learner.store.get("test:leaf")
        mid_state = credit_learner.store.get("test:mid")
        root_state = credit_learner.store.get("test:root")

        # Direct > parent > grandparent (decay)
        leaf_delta = leaf_state.alpha - 1.0
        mid_delta = mid_state.alpha - 1.0
        root_delta = root_state.alpha - 1.0

        assert leaf_delta > mid_delta > root_delta > 0

    def test_dag_build_failure_handled_gracefully(self, monkeypatch, tmp_path):
        from qortex.core.models import ConceptNode
        from qortex.mcp.server import _maybe_propagate_credit

        monkeypatch.setenv("QORTEX_CREDIT_PROPAGATION", "on")
        reset_flags()

        nodes = {
            "test:x": ConceptNode(id="test:x", name="X", description="", domain="test", source_id="test"),
        }
        mock_backend = MagicMock()
        mock_backend.get_node.side_effect = lambda nid, **kw: nodes.get(nid)

        import qortex.mcp.server as srv
        monkeypatch.setattr(srv, "_backend", mock_backend)
        monkeypatch.setattr(srv, "_learning_state_dir", str(tmp_path))
        monkeypatch.setattr(srv, "_learners", {})

        # Make from_backend raise
        with patch(
            "qortex.causal.dag.CausalDAG.from_backend",
            side_effect=RuntimeError("boom"),
        ):
            result = _maybe_propagate_credit("q1", {"test:x": "accepted"})

        # Should return None (graceful degradation), not crash
        assert result is None


class TestCreditPropagatedEvent:
    """Verify CreditPropagated event is emitted and captured."""

    def test_event_in_jsonl_all_events(self):
        from qortex.observability.events import CreditPropagated
        from qortex.observability.subscribers.jsonl import _ALL_EVENTS

        assert CreditPropagated in _ALL_EVENTS

    def test_structlog_handler_logs_credit_propagated(self, caplog):
        """Verify structlog handler actually logs credit.propagated with fields."""
        import logging
        import time

        from qortex.observability.config import ObservabilityConfig
        from qortex.observability.emitter import configure, emit, reset
        from qortex.observability.events import CreditPropagated

        reset()
        configure(ObservabilityConfig(
            log_formatter="stdlib",
            log_destination="stderr",
            log_level="DEBUG",
            log_format="json",
        ))

        with caplog.at_level(logging.DEBUG, logger="qortex.events"):
            emit(CreditPropagated(
                query_id="q1",
                concept_count=5,
                direct_count=2,
                ancestor_count=3,
                total_alpha_delta=1.5,
                total_beta_delta=0.3,
                learner="credit",
            ))
            # pyventus dispatches async — give handler time to fire
            time.sleep(0.2)

        assert any("credit.propagated" in r.message for r in caplog.records)
        reset()

    def test_event_written_to_jsonl(self, tmp_path):
        import json

        from qortex.observability.emitter import configure, emit, reset
        from qortex.observability.events import CreditPropagated
        from qortex.observability.config import ObservabilityConfig

        reset()
        cfg = ObservabilityConfig(
            log_destination="stderr",
            jsonl_path=str(tmp_path / "events.jsonl"),
        )
        configure(cfg)

        emit(CreditPropagated(
            query_id="q1",
            concept_count=5,
            direct_count=2,
            ancestor_count=3,
            total_alpha_delta=1.5,
            total_beta_delta=0.3,
            learner="credit",
        ))

        lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        event_names = [e["event"] for e in events]
        assert "CreditPropagated" in event_names

        credit_event = next(e for e in events if e["event"] == "CreditPropagated")
        assert credit_event["concept_count"] == 5
        assert credit_event["direct_count"] == 2
        assert credit_event["ancestor_count"] == 3
        reset()
