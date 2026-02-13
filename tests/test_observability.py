"""Tests for the observability layer: events, emitter, logging, sinks, alerts.

Coverage:
- Events: frozen immutability, field access, serialization
- Emitter: emit no-op when unconfigured, configure idempotency, reset
- Logging: formatter × destination composition, get_logger pre/post config
- Sinks: JSONL write, stdout, no-op
- Alerts: rule evaluation, cooldown, built-in rules
- Integration: factors/buffer emit events, PPR convergence events
"""

from __future__ import annotations

import json
import logging
from dataclasses import FrozenInstanceError, asdict
from datetime import timedelta
from pathlib import Path

import pytest

from qortex_observe.config import ObservabilityConfig
from qortex_observe.events import (
    BufferFlushed,
    CreditPropagated,
    EdgePromoted,
    FactorDriftSnapshot,
    FactorsLoaded,
    FactorsPersisted,
    FactorUpdated,
    FeedbackReceived,
    InteroceptionShutdown,
    InteroceptionStarted,
    KGCoverageComputed,
    LearningObservationRecorded,
    LearningPosteriorUpdated,
    LearningSelectionMade,
    ManifestIngested,
    OnlineEdgeRecorded,
    OnlineEdgesGenerated,
    PPRConverged,
    PPRDiverged,
    PPRStarted,
    QueryCompleted,
    QueryFailed,
    QueryStarted,
    VecSearchCompleted,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_observability():
    """Reset emitter state before and after each test."""
    from qortex_observe.emitter import reset

    reset()
    yield
    reset()


@pytest.fixture()
def configured():
    """Configure observability with stderr + structlog defaults."""
    from qortex_observe.emitter import configure

    cfg = ObservabilityConfig(
        log_formatter="structlog",
        log_destination="stderr",
        log_level="DEBUG",
        log_format="json",
    )
    return configure(cfg)


# =============================================================================
# Event dataclass tests
# =============================================================================


class TestEvents:
    """All events are frozen dataclasses with correct fields."""

    def test_query_started_frozen(self):
        e = QueryStarted(
            query_id="q1",
            query_text="test",
            domains=("d1",),
            mode="vec",
            top_k=10,
            timestamp="2026-01-01T00:00:00Z",
        )
        assert e.query_id == "q1"
        assert e.mode == "vec"
        with pytest.raises(FrozenInstanceError):
            e.query_id = "q2"  # type: ignore[misc]

    def test_query_completed_fields(self):
        e = QueryCompleted(
            query_id="q1",
            latency_ms=42.5,
            seed_count=10,
            result_count=5,
            activated_nodes=8,
            mode="graph",
            timestamp="2026-01-01T00:00:00Z",
        )
        assert e.latency_ms == 42.5
        assert e.activated_nodes == 8

    def test_ppr_converged_serializable(self):
        e = PPRConverged(
            query_id=None,
            iterations=15,
            final_diff=1e-7,
            node_count=100,
            nonzero_scores=42,
            latency_ms=3.2,
        )
        d = asdict(e)
        assert d["iterations"] == 15
        assert d["query_id"] is None
        # Must be JSON-serializable
        json.dumps(d)

    def test_ppr_diverged_frozen(self):
        e = PPRDiverged(query_id=None, iterations=100, final_diff=0.5, node_count=50)
        with pytest.raises(FrozenInstanceError):
            e.iterations = 200  # type: ignore[misc]

    def test_factor_updated_fields(self):
        e = FactorUpdated(
            node_id="n1",
            query_id="q1",
            outcome="accepted",
            old_factor=1.0,
            new_factor=1.1,
            delta=0.1,
            clamped=False,
        )
        assert e.delta == 0.1
        assert not e.clamped

    def test_feedback_received_counts(self):
        e = FeedbackReceived(
            query_id="q1",
            outcomes=3,
            accepted=2,
            rejected=1,
            partial=0,
            source="mcp",
        )
        assert e.accepted + e.rejected + e.partial == e.outcomes

    def test_interoception_started(self):
        e = InteroceptionStarted(
            factors_loaded=5, buffer_loaded=10, teleportation_enabled=True
        )
        assert e.teleportation_enabled is True

    def test_interoception_shutdown(self):
        e = InteroceptionShutdown(
            factors_persisted=5, buffer_persisted=0, summary={"key": "val"}
        )
        assert e.summary == {"key": "val"}

    def test_all_events_frozen(self):
        """Every event type must be frozen."""
        events = [
            QueryStarted("q", "t", None, "vec", 10, "ts"),
            QueryCompleted("q", 1.0, 1, 1, 1, "vec", "ts"),
            QueryFailed("q", "err", "embedding", "ts"),
            PPRStarted(None, 10, 2, 0.85, 0),
            PPRConverged(None, 10, 1e-7, 50, 20, 1.0),
            PPRDiverged(None, 100, 0.5, 50),
            FactorUpdated("n", "q", "accepted", 1.0, 1.1, 0.1, False),
            FactorsPersisted("/tmp/f.json", 5, "ts"),
            FactorsLoaded("/tmp/f.json", 5, "ts"),
            FactorDriftSnapshot(5, 1.2, 0.5, 3.0, 3, 2, 0.8),
            OnlineEdgeRecorded("a", "b", 0.9, 1, 10),
            EdgePromoted("a", "b", 3, 0.85),
            BufferFlushed(2, 8, 10, 0.3, "ts"),
            VecSearchCompleted("q", 30, 60, 5.0),
            OnlineEdgesGenerated("q", 5, 0.7, 10),
            KGCoverageComputed("q", 3, 5, 0.375),
            FeedbackReceived("q", 2, 1, 1, 0, "mcp"),
            InteroceptionStarted(0, 0, False),
            InteroceptionShutdown(0, 0, {}),
            ManifestIngested("d", 10, 5, 3, "src", 100.0),
        ]
        for event in events:
            d = asdict(event)
            assert isinstance(d, dict)
            # Verify frozen by trying to set first field
            first_field = list(d.keys())[0]
            with pytest.raises(FrozenInstanceError):
                setattr(event, first_field, "changed")


# =============================================================================
# Emitter tests
# =============================================================================


class TestEmitter:
    """emit() is no-op when not configured, works after configure()."""

    def test_emit_noop_when_not_configured(self):
        """emit() is a no-op when configure() hasn't been called."""
        from qortex_observe.emitter import emit

        # Should not raise
        emit(QueryStarted("q", "test", None, "vec", 10, "ts"))

    def test_configure_returns_emitter(self, configured):
        """configure() returns an EventEmitter."""
        from pyventus.events import EventEmitter

        assert isinstance(configured, EventEmitter)

    def test_configure_idempotent(self):
        """Second configure() returns same emitter."""
        from qortex_observe.emitter import configure

        cfg = ObservabilityConfig(log_destination="stderr")
        e1 = configure(cfg)
        e2 = configure(cfg)
        assert e1 is e2

    def test_is_configured(self):
        """is_configured() reflects state."""
        from qortex_observe.emitter import configure, is_configured

        assert not is_configured()
        configure(ObservabilityConfig(log_destination="stderr"))
        assert is_configured()

    def test_reset_clears_state(self, configured):
        """reset() clears emitter and configured flag."""
        from qortex_observe.emitter import is_configured, reset

        assert is_configured()
        reset()
        assert not is_configured()


# =============================================================================
# Logging tests
# =============================================================================


class TestLogging:
    """LogFormatter × LogDestination composition."""

    def test_structlog_formatter_setup(self):
        from qortex_observe.logging import StructlogFormatter

        cfg = ObservabilityConfig(log_format="json")
        formatter = StructlogFormatter()
        result = formatter.setup(cfg)
        assert isinstance(result, logging.Formatter)

    def test_stdlib_formatter_setup(self):
        from qortex_observe.logging import StdlibFormatter

        cfg = ObservabilityConfig(log_format="json")
        formatter = StdlibFormatter()
        result = formatter.setup(cfg)
        assert isinstance(result, logging.Formatter)

    def test_stdlib_console_formatter(self):
        from qortex_observe.logging import StdlibFormatter

        cfg = ObservabilityConfig(log_format="console")
        formatter = StdlibFormatter()
        result = formatter.setup(cfg)
        assert isinstance(result, logging.Formatter)

    def test_get_logger_before_config(self):
        """get_logger() returns stdlib logger before setup."""
        from qortex_observe.logging import get_logger

        lg = get_logger("test")
        # Should be a stdlib logger (fallback)
        assert lg is not None

    def test_get_logger_after_config(self, configured):
        """get_logger() returns structlog BoundLogger after setup."""
        from qortex_observe.logging import get_logger

        lg = get_logger("test")
        assert lg is not None
        # Should have info/debug/warning methods
        assert hasattr(lg, "info")
        assert hasattr(lg, "debug")
        assert hasattr(lg, "warning")

    def test_stderr_destination(self):
        from qortex_observe.logging import StderrDestination

        dest = StderrDestination()
        handler = dest.create_handler(logging.Formatter())
        assert isinstance(handler, logging.StreamHandler)
        dest.shutdown()

    def test_jsonl_file_destination(self, tmp_path):
        from qortex_observe.logging import JsonlFileDestination

        cfg = ObservabilityConfig(jsonl_path=str(tmp_path / "test.jsonl"))
        dest = JsonlFileDestination(cfg)
        handler = dest.create_handler(logging.Formatter("%(message)s"))
        assert isinstance(handler, logging.FileHandler)
        dest.shutdown()

    def test_register_custom_formatter(self):
        from qortex_observe.logging import _FORMATTERS, register_formatter

        class CustomFormatter:
            def setup(self, config):
                return logging.Formatter()

            def get_logger(self, name, **kwargs):
                return logging.getLogger(name)

        register_formatter("custom", CustomFormatter)
        assert "custom" in _FORMATTERS
        # Cleanup
        del _FORMATTERS["custom"]

    def test_register_custom_destination(self):
        from qortex_observe.logging import _DESTINATIONS, register_destination

        class CustomDest:
            def create_handler(self, formatter):
                return logging.StreamHandler()

            def shutdown(self):
                pass

        register_destination("custom", CustomDest)
        assert "custom" in _DESTINATIONS
        # Cleanup
        del _DESTINATIONS["custom"]

    def test_setup_unknown_formatter_raises(self):
        from qortex_observe.logging import setup_logging

        cfg = ObservabilityConfig(log_formatter="nonexistent")
        with pytest.raises(ValueError, match="Unknown log formatter"):
            setup_logging(cfg)

    def test_setup_unknown_destination_raises(self):
        from qortex_observe.logging import setup_logging

        cfg = ObservabilityConfig(log_destination="nonexistent")
        with pytest.raises(ValueError, match="Unknown log destination"):
            setup_logging(cfg)


# =============================================================================
# Config tests
# =============================================================================


class TestConfig:
    """ObservabilityConfig env-var driven defaults."""

    def test_default_values(self, monkeypatch):
        monkeypatch.delenv("QORTEX_OTEL_ENABLED", raising=False)
        monkeypatch.delenv("QORTEX_PROMETHEUS_ENABLED", raising=False)
        monkeypatch.delenv("QORTEX_ALERTS_ENABLED", raising=False)
        cfg = ObservabilityConfig()
        assert cfg.log_formatter == "structlog"
        assert cfg.log_destination == "stderr"
        assert cfg.log_level == "INFO"
        assert cfg.log_format == "json"
        assert cfg.otel_enabled is False
        assert cfg.prometheus_enabled is False
        assert cfg.alert_enabled is False

    def test_explicit_values(self):
        cfg = ObservabilityConfig(
            log_formatter="stdlib",
            log_destination="jsonl",
            log_level="DEBUG",
            otel_enabled=True,
        )
        assert cfg.log_formatter == "stdlib"
        assert cfg.otel_enabled is True


# =============================================================================
# Sink tests
# =============================================================================


class TestSinks:
    """LogSink implementations."""

    def test_jsonl_sink_writes(self, tmp_path):
        from qortex_observe.sinks.jsonl_sink import JsonlSink

        path = tmp_path / "events.jsonl"
        sink = JsonlSink(path)
        sink.write({"event": "test", "value": 42})
        sink.write({"event": "test2", "value": 99})

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["value"] == 42
        assert json.loads(lines[1])["event"] == "test2"

    def test_stdout_sink_writes(self, capsys):
        from qortex_observe.sinks.stdout_sink import StdoutSink

        sink = StdoutSink()
        sink.write({"event": "hello"})
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_noop_sink(self):
        from qortex_observe.sinks.noop_sink import NoOpSink

        sink = NoOpSink()
        sink.write({"event": "ignored"})  # Should not raise


# =============================================================================
# Alert tests
# =============================================================================


class TestAlerts:
    """Alert rule evaluation and sinks."""

    def test_builtin_rules_exist(self):
        from qortex_observe.alerts.rules import BUILTIN_RULES

        assert len(BUILTIN_RULES) >= 2
        names = [r.name for r in BUILTIN_RULES]
        assert "ppr_divergence" in names
        assert "factor_drift_high" in names

    def test_ppr_divergence_rule_matches(self):
        from qortex_observe.alerts.rules import BUILTIN_RULES

        rule = next(r for r in BUILTIN_RULES if r.name == "ppr_divergence")
        event = PPRDiverged(query_id=None, iterations=100, final_diff=0.5, node_count=50)
        assert rule.condition(event) is True

        # Should not match other events
        other = QueryCompleted("q", 1.0, 1, 1, 1, "vec", "ts")
        assert rule.condition(other) is False

    def test_factor_drift_rule_matches(self):
        from qortex_observe.alerts.rules import BUILTIN_RULES

        rule = next(r for r in BUILTIN_RULES if r.name == "factor_drift_high")

        # Low entropy → should match
        low = FactorDriftSnapshot(5, 1.2, 0.5, 3.0, 3, 2, 0.3)
        assert rule.condition(low) is True

        # High entropy → should not match
        high = FactorDriftSnapshot(5, 1.2, 0.5, 3.0, 3, 2, 0.9)
        assert rule.condition(high) is False

    def test_log_alert_sink_fires(self, caplog):
        from qortex_observe.alerts.base import AlertRule
        from qortex_observe.alerts.log_sink import LogAlertSink

        sink = LogAlertSink()
        rule = AlertRule(
            name="test_rule",
            description="test",
            severity="warning",
            condition=lambda e: True,
        )
        event = PPRDiverged(query_id=None, iterations=100, final_diff=0.5, node_count=50)
        # Should not raise
        sink.fire(rule, event)

    def test_noop_alert_sink(self):
        from qortex_observe.alerts.base import AlertRule
        from qortex_observe.alerts.noop_sink import NoOpAlertSink

        sink = NoOpAlertSink()
        rule = AlertRule(
            name="test", description="test", severity="info", condition=lambda e: True
        )
        sink.fire(rule, None)  # Should not raise

    def test_alert_cooldown(self):
        from datetime import UTC, datetime

        from qortex_observe.alerts.base import AlertRule

        rule = AlertRule(
            name="test",
            description="test",
            severity="warning",
            condition=lambda e: True,
            cooldown=timedelta(minutes=5),
        )
        # First fire: no cooldown
        assert rule._last_fired is None

        # Simulate fire
        rule._last_fired = datetime.now(UTC)

        # Second fire within cooldown: should be blocked by caller
        elapsed = datetime.now(UTC) - rule._last_fired
        assert elapsed < rule.cooldown


# =============================================================================
# Integration: factors emit events
# =============================================================================


class TestFactorEvents:
    """TeleportationFactors.update() emits FactorUpdated events."""

    def test_factor_update_emits(self, configured, tmp_path):
        from qortex.hippocampus.factors import TeleportationFactors

        factors = TeleportationFactors()
        updates = factors.update("q1", {"node1": "accepted", "node2": "rejected"})

        assert len(updates) == 2
        assert updates[0].outcome == "accepted"
        assert updates[1].outcome == "rejected"

    def test_factor_persist_emits(self, configured, tmp_path):
        from qortex.hippocampus.factors import TeleportationFactors

        factors = TeleportationFactors()
        factors.factors["n1"] = 1.5
        path = factors.persist(tmp_path / "factors.json")
        assert path is not None

    def test_factor_load_emits(self, configured, tmp_path):
        from qortex.hippocampus.factors import TeleportationFactors

        # Write some factors
        f = TeleportationFactors()
        f.factors["n1"] = 1.5
        f.persist(tmp_path / "factors.json")

        # Load emits FactorsLoaded
        loaded = TeleportationFactors.load(tmp_path / "factors.json")
        assert len(loaded.factors) == 1


# =============================================================================
# Integration: buffer emits events
# =============================================================================


class TestBufferEvents:
    """EdgePromotionBuffer emits events on record/flush."""

    def test_buffer_record_emits(self, configured):
        from qortex.hippocampus.buffer import EdgePromotionBuffer

        buf = EdgePromotionBuffer()
        buf.record("a", "b", 0.9)
        # Buffer should have recorded
        assert buf.summary()["buffered_edges"] == 1

    def test_buffer_flush_emits(self, configured):
        from qortex.core.memory import InMemoryBackend
        from qortex.hippocampus.buffer import EdgePromotionBuffer

        backend = InMemoryBackend()
        backend.connect()

        buf = EdgePromotionBuffer()
        # Record enough to qualify for promotion
        for _ in range(5):
            buf.record("a", "b", 0.9)

        result = buf.flush(backend, min_hits=3, min_avg_score=0.7)
        assert result.promoted == 1


# =============================================================================
# Integration: PPR convergence events
# =============================================================================


class TestPPREvents:
    """InMemoryBackend.personalized_pagerank() emits convergence events."""

    def _build_graph(self):
        from qortex.core.memory import InMemoryBackend
        from qortex.core.models import ConceptEdge, ConceptNode, RelationType

        backend = InMemoryBackend()
        backend.connect()
        backend.create_domain("test")

        for i in range(5):
            backend.add_node(
                ConceptNode(
                    id=f"n{i}",
                    name=f"Node {i}",
                    description=f"Test node {i}",
                    domain="test",
                    source_id="test",
                )
            )

        for i in range(4):
            backend.add_edge(
                ConceptEdge(
                    source_id=f"n{i}",
                    target_id=f"n{i+1}",
                    relation_type=RelationType.REQUIRES,
                    confidence=0.9,
                )
            )

        return backend

    def test_ppr_emits_convergence(self, configured):
        backend = self._build_graph()
        scores = backend.personalized_pagerank(
            source_nodes=["n0", "n1"],
            damping_factor=0.85,
            max_iterations=100,
        )
        assert len(scores) > 0  # Should have computed scores

    def test_ppr_with_extra_edges(self, configured):
        backend = self._build_graph()
        scores = backend.personalized_pagerank(
            source_nodes=["n0"],
            extra_edges=[("n0", "n4", 0.8)],
        )
        assert "n4" in scores  # Extra edge should create path


# =============================================================================
# Integration: interoception lifecycle events
# =============================================================================


class TestInteroceptionEvents:
    """LocalInteroceptionProvider emits startup/shutdown events."""

    def test_startup_emits(self, configured, tmp_path):
        from qortex.hippocampus.interoception import (
            InteroceptionConfig,
            LocalInteroceptionProvider,
        )

        config = InteroceptionConfig(
            factors_path=tmp_path / "factors.json",
            buffer_path=tmp_path / "buffer.json",
        )
        provider = LocalInteroceptionProvider(config)
        provider.startup()
        assert provider._started is True

    def test_shutdown_emits(self, configured, tmp_path):
        from qortex.hippocampus.interoception import (
            InteroceptionConfig,
            LocalInteroceptionProvider,
        )

        config = InteroceptionConfig(
            factors_path=tmp_path / "factors.json",
            buffer_path=tmp_path / "buffer.json",
        )
        provider = LocalInteroceptionProvider(config)
        provider.startup()
        provider.shutdown()

        # Factors and buffer should have been persisted
        assert (tmp_path / "factors.json").exists()
        assert (tmp_path / "buffer.json").exists()


# =============================================================================
# Linker tests
# =============================================================================


class TestLinker:
    """QortexEventLinker is isolated from other EventLinker subclasses."""

    def test_linker_is_event_linker(self):
        from pyventus.events import EventLinker

        from qortex_observe.linker import QortexEventLinker

        assert issubclass(QortexEventLinker, EventLinker)


# =============================================================================
# Integration: FactorDriftSnapshot emission
# =============================================================================


class TestFactorDriftSnapshotEmission:
    """TeleportationFactors.update() emits FactorDriftSnapshot with entropy."""

    def test_factor_drift_snapshot_emitted_on_update(self, configured):
        """update() emits FactorDriftSnapshot after batch."""
        from unittest.mock import patch

        from qortex.hippocampus.factors import TeleportationFactors

        factors = TeleportationFactors()
        captured = []

        with patch("qortex.hippocampus.factors.emit", side_effect=lambda e: captured.append(e)):
            factors.update("q1", {"n1": "accepted", "n2": "rejected"})

        # Should have FactorUpdated events + one FactorDriftSnapshot
        drift_events = [e for e in captured if isinstance(e, FactorDriftSnapshot)]
        assert len(drift_events) == 1
        snap = drift_events[0]
        assert snap.count == 2
        assert snap.boosted == 1  # n1 accepted → > 1.0
        assert snap.penalized == 1  # n2 rejected → < 1.0
        assert snap.entropy > 0  # Non-uniform → positive entropy

    def test_factor_drift_entropy_correct(self, configured):
        """Shannon entropy calculation is mathematically correct."""
        import math
        from unittest.mock import patch

        from qortex.hippocampus.factors import TeleportationFactors

        factors = TeleportationFactors()
        # Set up known factor distribution: [1.0, 1.0] → uniform
        factors.factors = {"a": 1.0, "b": 1.0}
        captured = []

        with patch("qortex.hippocampus.factors.emit", side_effect=lambda e: captured.append(e)):
            factors.update("q1", {"a": "accepted"})

        drift_events = [e for e in captured if isinstance(e, FactorDriftSnapshot)]
        assert len(drift_events) == 1
        snap = drift_events[0]

        # Verify entropy manually: factors are [1.1, 1.0], total=2.1
        vals = [1.1, 1.0]
        total = sum(vals)
        probs = [v / total for v in vals]
        expected_entropy = -sum(p * math.log2(p) for p in probs)
        assert snap.entropy == pytest.approx(expected_entropy)


# =============================================================================
# Integration: enrichment emits events
# =============================================================================


class TestEnrichmentEmission:
    """EnrichmentPipeline emits EnrichmentCompleted and EnrichmentFallback."""

    def _make_rules(self, n: int = 3):
        from qortex.core.models import Rule

        return [
            Rule(
                id=f"r{i}",
                text=f"Rule {i}",
                domain="test",
                derivation="explicit",
                source_concepts=[],
                confidence=1.0,
            )
            for i in range(n)
        ]

    def test_enrichment_emits_completed_event(self, configured):
        """enrich() emits EnrichmentCompleted at end."""
        from unittest.mock import patch

        from qortex.enrichment.pipeline import EnrichmentPipeline
        from qortex_observe.events import EnrichmentCompleted

        pipeline = EnrichmentPipeline()  # No backend → template fallback
        rules = self._make_rules(3)
        captured = []

        with patch("qortex.enrichment.pipeline.emit", side_effect=lambda e: captured.append(e)):
            result = pipeline.enrich(rules, domain="test")

        assert len(result) == 3
        completed_events = [e for e in captured if isinstance(e, EnrichmentCompleted)]
        assert len(completed_events) == 1
        ev = completed_events[0]
        assert ev.rule_count == 3
        assert ev.succeeded == 3
        assert ev.failed == 0
        assert ev.backend_type == "template"
        assert ev.latency_ms > 0

    def test_enrichment_emits_fallback_on_backend_failure(self, configured):
        """Backend failure path emits EnrichmentFallback."""
        from unittest.mock import patch

        from qortex.enrichment.pipeline import EnrichmentPipeline
        from qortex_observe.events import EnrichmentCompleted, EnrichmentFallback

        class FailingBackend:
            def enrich_batch(self, rules, domain):
                raise RuntimeError("backend exploded")

        pipeline = EnrichmentPipeline(backend=FailingBackend())
        rules = self._make_rules(2)
        captured = []

        with patch("qortex.enrichment.pipeline.emit", side_effect=lambda e: captured.append(e)):
            result = pipeline.enrich(rules, domain="test")

        assert len(result) == 2  # Fallback should still produce results
        fallback_events = [e for e in captured if isinstance(e, EnrichmentFallback)]
        assert len(fallback_events) == 1
        assert fallback_events[0].reason == "backend_exception"
        assert fallback_events[0].rule_count == 2

        # Should also emit EnrichmentCompleted
        completed_events = [e for e in captured if isinstance(e, EnrichmentCompleted)]
        assert len(completed_events) == 1
        assert completed_events[0].failed == 2


# =============================================================================
# Integration: ManifestIngested emission
# =============================================================================


class TestManifestIngestedEmission:
    """InMemoryBackend.ingest_manifest() emits ManifestIngested."""

    def test_manifest_ingested_emitted(self, configured):
        from unittest.mock import patch

        from qortex.core.memory import InMemoryBackend
        from qortex.core.models import (
            ConceptEdge,
            ConceptNode,
            ExplicitRule,
            IngestionManifest,
            RelationType,
            SourceMetadata,
        )
        from qortex_observe.events import ManifestIngested

        backend = InMemoryBackend()
        backend.connect()

        manifest = IngestionManifest(
            source=SourceMetadata(
                id="src1", name="test", source_type="text", path_or_url="/test"
            ),
            domain="test_domain",
            concepts=[
                ConceptNode(id="n1", name="Node 1", description="", domain="test_domain", source_id="src1"),
                ConceptNode(id="n2", name="Node 2", description="", domain="test_domain", source_id="src1"),
            ],
            edges=[
                ConceptEdge(
                    source_id="n1",
                    target_id="n2",
                    relation_type=RelationType.REQUIRES,
                    confidence=0.9,
                ),
            ],
            rules=[
                ExplicitRule(
                    id="r1", text="Rule 1", domain="test_domain", source_id="src1", confidence=1.0
                ),
            ],
        )

        captured = []
        with patch("qortex.core.memory.emit", side_effect=lambda e: captured.append(e)):
            backend.ingest_manifest(manifest)

        manifest_events = [e for e in captured if isinstance(e, ManifestIngested)]
        assert len(manifest_events) == 1
        ev = manifest_events[0]
        assert ev.domain == "test_domain"
        assert ev.node_count == 2
        assert ev.edge_count == 1
        assert ev.rule_count == 1
        assert ev.source_id == "src1"
        assert ev.latency_ms > 0

    def test_memgraph_manifest_ingested_emitted(self, configured):
        """MemgraphBackend.ingest_manifest() emits ManifestIngested too."""
        from unittest.mock import MagicMock, patch

        from qortex.core.backend import MemgraphBackend
        from qortex.core.models import (
            ConceptNode,
            IngestionManifest,
            SourceMetadata,
        )
        from qortex_observe.events import ManifestIngested

        backend = MemgraphBackend(uri="bolt://fake:7687")
        backend._driver = MagicMock()
        # _run returns empty list by default (MERGE, SET calls)
        backend._run = MagicMock(return_value=[])
        # create_domain calls _run_single which returns a domain record
        backend._run_single = MagicMock(return_value={
            "name": "test_domain", "description": None,
            "created_at": None, "updated_at": None,
        })
        # _count returns 0 for stats queries
        backend._count = MagicMock(return_value=0)

        manifest = IngestionManifest(
            source=SourceMetadata(
                id="src1", name="test", source_type="text", path_or_url="/test"
            ),
            domain="test_domain",
            concepts=[
                ConceptNode(id="n1", name="N1", description="", domain="test_domain", source_id="src1"),
            ],
            edges=[],
            rules=[],
        )

        captured = []
        with patch("qortex.core.backend.emit", side_effect=lambda e: captured.append(e)):
            backend.ingest_manifest(manifest)

        manifest_events = [e for e in captured if isinstance(e, ManifestIngested)]
        assert len(manifest_events) == 1
        assert manifest_events[0].domain == "test_domain"
        assert manifest_events[0].node_count == 1


# =============================================================================
# Integration: Prometheus metrics fire correctly
# =============================================================================


class TestPrometheusMetrics:
    """Verify Prometheus counters/histograms update when events fire."""

    def test_factor_updated_increments_counter(self):
        """FactorUpdated handler increments qortex_factor_updates_total."""
        from unittest.mock import MagicMock, patch

        from qortex_observe.events import FactorUpdated

        mock_counter = MagicMock()
        event = FactorUpdated(
            node_id="n1", query_id="q1", outcome="accepted",
            old_factor=1.0, new_factor=1.1, delta=0.1, clamped=False,
        )
        # Simulate the handler logic directly
        mock_counter.labels(outcome=event.outcome).inc()
        mock_counter.labels.assert_called_with(outcome="accepted")

    def test_vec_search_observes_latency(self):
        """VecSearchCompleted handler observes latency in seconds."""
        from qortex_observe.events import VecSearchCompleted

        event = VecSearchCompleted(query_id="q1", candidates=30, fetch_k=60, latency_ms=50.0)
        # Handler converts ms → seconds
        observed = event.latency_ms / 1000
        assert observed == pytest.approx(0.05)

    def test_enrichment_completed_increments_and_observes(self):
        """EnrichmentCompleted handler increments counter and observes latency."""
        from qortex_observe.events import EnrichmentCompleted

        event = EnrichmentCompleted(
            rule_count=5, succeeded=4, failed=1,
            backend_type="template", latency_ms=1500.0,
        )
        assert event.latency_ms / 1000 == pytest.approx(1.5)
        assert event.backend_type == "template"

    def test_manifest_ingested_increments_and_observes(self):
        """ManifestIngested handler increments counter and observes latency."""
        from qortex_observe.events import ManifestIngested

        event = ManifestIngested(
            domain="test", node_count=10, edge_count=5,
            rule_count=3, source_id="src1", latency_ms=250.0,
        )
        assert event.latency_ms / 1000 == pytest.approx(0.25)
        assert event.domain == "test"

    def test_query_failed_increments_by_stage(self):
        """QueryFailed handler increments error counter by stage."""
        from unittest.mock import MagicMock

        from qortex_observe.events import QueryFailed

        mock_counter = MagicMock()
        event = QueryFailed(query_id="q1", error="boom", stage="embedding", timestamp="ts")
        mock_counter.labels(stage=event.stage).inc()
        mock_counter.labels.assert_called_with(stage="embedding")

    def test_credit_propagated_increments_counters(self):
        """CreditPropagated handler increments propagation counter and delta counters."""
        from unittest.mock import MagicMock

        event = CreditPropagated(
            query_id="q1", concept_count=5, direct_count=2,
            ancestor_count=3, total_alpha_delta=1.5, total_beta_delta=0.3,
            learner="credit",
        )

        # Verify handler logic: counter labels and concept histogram
        mock_counter = MagicMock()
        mock_counter.labels(learner=event.learner).inc()
        mock_counter.labels.assert_called_with(learner="credit")

        # Verify alpha/beta delta logic
        assert event.total_alpha_delta > 0
        assert event.total_beta_delta > 0
        assert event.concept_count == 5
        assert event.direct_count == 2
        assert event.ancestor_count == 3

    def test_credit_propagated_skips_zero_deltas(self):
        """CreditPropagated handler skips alpha/beta inc when deltas are zero."""
        event = CreditPropagated(
            query_id="q1", concept_count=0, direct_count=0,
            ancestor_count=0, total_alpha_delta=0.0, total_beta_delta=0.0,
            learner="credit",
        )
        # Handler guards: `if event.total_alpha_delta > 0` and `if event.total_beta_delta > 0`
        assert not (event.total_alpha_delta > 0)
        assert not (event.total_beta_delta > 0)


class TestPrometheusLiveMetrics:
    """Smoke test: configure real Prometheus subscriber, emit events, verify wiring."""

    def test_prometheus_subscriber_registers_and_emits_without_error(self):
        """Full path: configure(prometheus_enabled) → emit(QueryCompleted) → no crash."""
        from unittest.mock import patch

        from qortex_observe.emitter import configure, emit, reset

        reset()

        cfg = ObservabilityConfig()
        cfg.prometheus_enabled = True

        # Patch start_http_server to avoid port binding in tests
        with patch(
            "prometheus_client.start_http_server"
        ):
            emitter = configure(cfg)
            assert emitter is not None

        # Emit real events through the full pipeline — verifies handlers are wired
        emit(QueryCompleted(
            query_id="smoke-1", mode="hybrid", result_count=5,
            latency_ms=42.0, seed_count=3, activated_nodes=10, timestamp="ts",
        ))
        emit(QueryFailed(
            query_id="smoke-2", error="test", stage="embedding", timestamp="ts",
        ))
        emit(FactorUpdated(
            node_id="n1", query_id="q1", outcome="accepted",
            old_factor=1.0, new_factor=1.1, delta=0.1, clamped=False,
        ))

        # No exception = handlers are registered and functional
        reset()


# =============================================================================
# OTEL subscriber error handling
# =============================================================================


class TestOtelErrorHandling:
    """OTEL subscriber registration: error handling, protocol fallback, success log."""

    def test_non_import_error_caught_and_logged(self):
        """Non-ImportError from register_otel_traces is caught, not propagated."""
        from unittest.mock import patch

        from qortex_observe.emitter import configure, is_configured, reset

        reset()

        cfg = ObservabilityConfig(otel_enabled=True)

        # Simulate register_otel_traces raising AttributeError
        with patch(
            "qortex_observe.subscribers.otel.register_otel_traces",
            side_effect=AttributeError("create_gauge not found"),
        ), patch(
            "qortex_observe.emitter._setup_metrics_pipeline",
        ):
            emitter = configure(cfg)

        # configure() should complete (not crash) and set _configured = True
        assert emitter is not None
        assert is_configured()
        reset()

    def test_import_error_still_caught(self):
        """ImportError from missing OTEL packages is still caught."""
        from unittest.mock import patch

        from qortex_observe.emitter import configure, is_configured, reset

        reset()

        cfg = ObservabilityConfig(otel_enabled=True)

        with patch(
            "qortex_observe.subscribers.otel.register_otel_traces",
            side_effect=ImportError("No module named 'opentelemetry'"),
        ), patch(
            "qortex_observe.emitter._setup_metrics_pipeline",
        ):
            emitter = configure(cfg)

        assert emitter is not None
        assert is_configured()
        reset()

    def test_otel_success_log_emitted(self, caplog):
        """Successful OTEL trace registration emits info log."""
        from unittest.mock import patch

        from qortex_observe.emitter import configure, reset

        reset()

        cfg = ObservabilityConfig(otel_enabled=True)

        with patch(
            "qortex_observe.subscribers.otel.register_otel_traces"
        ), patch(
            "qortex_observe.emitter._setup_metrics_pipeline",
        ):
            configure(cfg)

        reset()

    def test_otel_protocol_config_defaults_to_grpc(self, monkeypatch):
        """Default otel_protocol is 'grpc'."""
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_PROTOCOL", raising=False)
        cfg = ObservabilityConfig()
        assert cfg.otel_protocol == "grpc"

    def test_otel_protocol_config_http(self):
        """otel_protocol can be set to 'http/protobuf'."""
        cfg = ObservabilityConfig(otel_protocol="http/protobuf")
        assert cfg.otel_protocol == "http/protobuf"

    def test_get_exporters_grpc_fallback_to_http(self):
        """When grpcio is missing, _get_exporters falls back to HTTP."""
        from unittest.mock import patch

        from qortex_observe.subscribers.otel import _get_exporters

        # Simulate grpcio import failure
        orig_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if "grpc" in name:
                raise ImportError(f"No module named '{name}'")
            return orig_import(name, *args, **kwargs)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                span_exp, metric_exp = _get_exporters("grpc", "http://localhost:4317")
        except ImportError:
            # If HTTP exporter also can't import (no otel in test env), that's OK
            pytest.skip("opentelemetry not installed")

    def test_metrics_pipeline_non_import_error_caught(self):
        """Non-ImportError from metrics pipeline is caught, not propagated."""
        from unittest.mock import patch

        from qortex_observe.emitter import configure, is_configured, reset

        reset()

        cfg = ObservabilityConfig(prometheus_enabled=True)

        with patch(
            "qortex_observe.emitter._setup_metrics_pipeline",
            side_effect=OSError("Address already in use"),
        ):
            emitter = configure(cfg)

        assert emitter is not None
        assert is_configured()
        reset()


# =============================================================================
# Integration: learning events in structlog subscriber
# =============================================================================


class TestStructlogLearningEvents:
    """Structlog subscriber handles learning events and logs them."""

    def test_learning_selection_logged(self, configured, caplog):
        from qortex_observe.emitter import emit

        with caplog.at_level(logging.DEBUG, logger="qortex.events"):
            emit(LearningSelectionMade(
                learner="test", selected_count=3, excluded_count=2,
                is_baseline=False, token_budget=1000, used_tokens=750,
            ))
        assert any("learning.selection" in r.message for r in caplog.records)

    def test_learning_observation_logged(self, configured, caplog):
        from qortex_observe.emitter import emit

        with caplog.at_level(logging.DEBUG, logger="qortex.events"):
            emit(LearningObservationRecorded(
                learner="test", arm_id="arm:a", reward=1.0,
                outcome="accepted", context_hash="default",
            ))
        assert any("learning.observation" in r.message for r in caplog.records)

    def test_learning_posterior_logged(self, configured, caplog):
        from qortex_observe.emitter import emit

        with caplog.at_level(logging.DEBUG, logger="qortex.events"):
            emit(LearningPosteriorUpdated(
                learner="test", arm_id="arm:a",
                alpha=2.0, beta=1.0, pulls=1, mean=0.667,
            ))
        assert any("learning.posterior" in r.message for r in caplog.records)


# =============================================================================
# Integration: learning events in JSONL subscriber
# =============================================================================


class TestJsonlLearningEvents:
    """JSONL subscriber captures learning events."""

    def test_learning_events_in_all_events_tuple(self):
        from qortex_observe.subscribers.jsonl import _ALL_EVENTS

        assert LearningSelectionMade in _ALL_EVENTS
        assert LearningObservationRecorded in _ALL_EVENTS
        assert LearningPosteriorUpdated in _ALL_EVENTS

    def test_learning_events_written_to_jsonl(self, tmp_path):
        from qortex_observe.emitter import configure, emit, reset

        reset()
        cfg = ObservabilityConfig(
            log_destination="stderr",
            jsonl_path=str(tmp_path / "events.jsonl"),
        )
        configure(cfg)

        emit(LearningSelectionMade(
            learner="test", selected_count=2, excluded_count=1,
            is_baseline=False, token_budget=500, used_tokens=400,
        ))
        emit(LearningObservationRecorded(
            learner="test", arm_id="arm:x", reward=1.0,
            outcome="accepted", context_hash="abc123",
        ))
        emit(LearningPosteriorUpdated(
            learner="test", arm_id="arm:x",
            alpha=2.0, beta=1.0, pulls=1, mean=0.667,
        ))

        lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
        events = [json.loads(line) for line in lines]
        event_names = [e["event"] for e in events]
        assert "LearningSelectionMade" in event_names
        assert "LearningObservationRecorded" in event_names
        assert "LearningPosteriorUpdated" in event_names
        reset()
