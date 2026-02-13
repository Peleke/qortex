"""Tests for config_snapshot_hash() and config hash on spans."""

from __future__ import annotations

import pytest
from qortex_observe.snapshot import config_snapshot_hash


class TestConfigSnapshotHash:
    """Blake2b-128 deterministic config hashing."""

    def test_deterministic(self):
        """Same config produces same hash across calls."""
        cfg = {"template_a": {"weight": 0.5}}
        h1 = config_snapshot_hash(rule_templates=cfg)
        h2 = config_snapshot_hash(rule_templates=cfg)
        assert h1 == h2

    def test_different_configs_different_hash(self):
        """Changing any config value changes the hash."""
        h1 = config_snapshot_hash(rule_templates={"a": 1})
        h2 = config_snapshot_hash(rule_templates={"a": 2})
        assert h1 != h2

    def test_sort_order_irrelevant(self):
        """Dict key order doesn't affect hash."""
        h1 = config_snapshot_hash(rule_templates={"b": 2, "a": 1})
        h2 = config_snapshot_hash(rule_templates={"a": 1, "b": 2})
        assert h1 == h2

    def test_none_configs_excluded(self):
        """None args don't affect hash."""
        h_empty = config_snapshot_hash()
        h_none = config_snapshot_hash(rule_templates=None, enrichment_config=None)
        assert h_empty == h_none

    def test_non_serializable_fallback(self):
        """Non-JSON-serializable values use str() fallback."""
        from datetime import datetime

        h = config_snapshot_hash(enrichment_config={"updated": datetime(2024, 1, 1)})
        assert isinstance(h, str)
        assert len(h) == 32

    def test_hash_length(self):
        """Blake2b-128 produces 32-char hex string."""
        h = config_snapshot_hash(rule_templates={"x": 1})
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)

    def test_empty_payload(self):
        """Empty config (no args) still produces valid hash."""
        h = config_snapshot_hash()
        assert isinstance(h, str)
        assert len(h) == 32

    def test_multiple_config_sections(self):
        """Hash includes all provided config sections."""
        h1 = config_snapshot_hash(rule_templates={"a": 1})
        h2 = config_snapshot_hash(rule_templates={"a": 1}, learner_configs={"b": 2})
        assert h1 != h2


class TestConfigHashOnSpans:
    """config.snapshot_hash attribute on traced spans."""

    @pytest.fixture(autouse=True)
    def _reset_tracer_provider(self):
        yield
        try:
            from opentelemetry import trace

            if hasattr(trace, "_TRACER_PROVIDER_SET_ONCE"):
                trace._TRACER_PROVIDER_SET_ONCE._done = False
        except ImportError:
            pass

    def test_span_has_config_hash_when_set(self):
        """Span includes config.snapshot_hash when contextvar is set."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from qortex_observe.tracing import _config_hash, traced

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        test_hash = config_snapshot_hash(rule_templates={"test": True})
        token = _config_hash.set(test_hash)

        try:
            @traced("test.op")
            def do_work():
                return 42

            do_work()
            provider.force_flush()
            spans = exporter.get_finished_spans()
            assert len(spans) == 1
            attrs = dict(spans[0].attributes)
            assert "config.snapshot_hash" in attrs
            assert attrs["config.snapshot_hash"] == test_hash
        finally:
            _config_hash.reset(token)

    def test_span_no_config_hash_when_unset(self):
        """Span omits config.snapshot_hash when contextvar is None."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from qortex_observe.tracing import traced

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        @traced("test.op")
        def do_work():
            return 42

        do_work()
        provider.force_flush()
        spans = exporter.get_finished_spans()
        assert len(spans) == 1
        attrs = dict(spans[0].attributes)
        assert "config.snapshot_hash" not in attrs

    def test_child_spans_inherit_config_hash(self):
        """Child spans also get config hash from contextvar."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        from qortex_observe.tracing import _config_hash, traced

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        from opentelemetry import trace

        trace.set_tracer_provider(provider)

        test_hash = config_snapshot_hash(learner_configs={"beta": True})
        token = _config_hash.set(test_hash)

        try:
            @traced("child.op")
            def child():
                return "ok"

            @traced("parent.op")
            def parent():
                return child()

            parent()
            provider.force_flush()
            spans = exporter.get_finished_spans()
            assert len(spans) == 2
            for s in spans:
                attrs = dict(s.attributes)
                assert attrs.get("config.snapshot_hash") == test_hash
        finally:
            _config_hash.reset(token)
