"""Exhaustive tests for the enrichment system."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from qortex.core.models import Rule
from qortex.enrichment.base import EnrichmentBackend
from qortex.enrichment.pipeline import (
    EnrichmentPipeline,
    TemplateEnrichmentFallback,
)
from qortex.projectors.models import EnrichedRule, RuleEnrichment

# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


def _make_rule(id: str = "r1", domain: str = "test") -> Rule:
    return Rule(
        id=id,
        text=f"Rule {id}",
        domain=domain,
        derivation="explicit",
        source_concepts=["c1"],
        confidence=0.9,
    )


def _make_rules(n: int, domain: str = "test") -> list[Rule]:
    return [_make_rule(f"r{i}", domain) for i in range(n)]


# -------------------------------------------------------------------------
# Template Enrichment Fallback
# -------------------------------------------------------------------------


class TestTemplateEnrichmentFallback:
    def test_enrich_batch(self):
        fb = TemplateEnrichmentFallback()
        rules = _make_rules(3)
        enrichments = fb.enrich_batch(rules, "test")

        assert len(enrichments) == 3
        for e in enrichments:
            assert isinstance(e, RuleEnrichment)
            assert e.enrichment_source == "template"
            assert e.enrichment_version == 1
            assert "test" in e.tags

    def test_enrich_batch_includes_category_tag(self):
        fb = TemplateEnrichmentFallback()
        rule = _make_rule()
        rule.category = "architectural"
        enrichments = fb.enrich_batch([rule], "test")
        assert "architectural" in enrichments[0].tags

    def test_enrich_batch_derived_rule_tag(self):
        fb = TemplateEnrichmentFallback()
        rule = Rule(
            id="r1",
            text="Derived",
            domain="test",
            derivation="derived",
            source_concepts=["c1"],
            confidence=0.8,
        )
        enrichments = fb.enrich_batch([rule], "test")
        assert "derived" in enrichments[0].tags

    def test_re_enrich(self):
        fb = TemplateEnrichmentFallback()
        existing = RuleEnrichment(
            context="Original context",
            antipattern="Original antipattern",
            rationale="Original rationale",
            tags=["original"],
            enrichment_version=1,
            source_contexts=["ctx1"],
        )
        result = fb.re_enrich(_make_rule(), existing, "new context info")
        assert result.enrichment_version == 2
        assert "new context info" in result.rationale
        assert "ctx1" in result.source_contexts
        assert "new context info" in result.source_contexts


# -------------------------------------------------------------------------
# Mock Enrichment Backend
# -------------------------------------------------------------------------


class MockEnrichmentBackend:
    """Test double that produces predictable enrichments."""

    def __init__(self, fail: bool = False):
        self._fail = fail
        self.calls: list[tuple] = []

    def enrich_batch(self, rules: list[Rule], domain: str) -> list[RuleEnrichment]:
        self.calls.append(("enrich_batch", len(rules), domain))
        if self._fail:
            raise RuntimeError("API failed")
        return [
            RuleEnrichment(
                context=f"Context for {r.id}",
                antipattern=f"Anti for {r.id}",
                rationale=f"Rationale for {r.id}",
                tags=[domain, r.id],
                enrichment_version=1,
                enriched_at=datetime.now(UTC),
                enrichment_source="mock",
            )
            for r in rules
        ]

    def re_enrich(self, rule: Rule, existing: RuleEnrichment, new_context: str) -> RuleEnrichment:
        self.calls.append(("re_enrich", rule.id, new_context))
        if self._fail:
            raise RuntimeError("API failed")
        return RuleEnrichment(
            context=f"Updated: {existing.context}",
            antipattern=existing.antipattern,
            rationale=f"{existing.rationale}. Also: {new_context}",
            tags=existing.tags,
            enrichment_version=existing.enrichment_version + 1,
            enriched_at=datetime.now(UTC),
            enrichment_source="mock",
            source_contexts=[*existing.source_contexts, new_context],
        )


class TestProtocolCompliance:
    def test_mock_backend_implements_protocol(self):
        assert isinstance(MockEnrichmentBackend(), EnrichmentBackend)

    def test_template_fallback_implements_protocol(self):
        assert isinstance(TemplateEnrichmentFallback(), EnrichmentBackend)


# -------------------------------------------------------------------------
# Enrichment Pipeline
# -------------------------------------------------------------------------


class TestEnrichmentPipeline:
    def test_enrich_with_backend(self):
        backend = MockEnrichmentBackend()
        pipeline = EnrichmentPipeline(backend=backend)
        rules = _make_rules(5)

        result = pipeline.enrich(rules, "test")

        assert len(result) == 5
        assert all(isinstance(r, EnrichedRule) for r in result)
        assert all(r.enrichment is not None for r in result)
        assert result[0].enrichment.context == "Context for r0"
        assert pipeline.stats.succeeded == 5
        assert pipeline.stats.failed == 0

    def test_enrich_without_backend_uses_fallback(self):
        pipeline = EnrichmentPipeline()  # No backend
        rules = _make_rules(3)

        result = pipeline.enrich(rules, "test")

        assert len(result) == 3
        assert all(r.enrichment is not None for r in result)
        assert all(r.enrichment.enrichment_source == "template" for r in result)

    def test_enrich_backend_failure_falls_back(self):
        backend = MockEnrichmentBackend(fail=True)
        pipeline = EnrichmentPipeline(backend=backend)
        rules = _make_rules(3)

        result = pipeline.enrich(rules, "test")

        assert len(result) == 3
        # Fell back to template
        assert all(r.enrichment.enrichment_source == "template" for r in result)
        assert pipeline.stats.failed == 3

    def test_enrich_empty_rules(self):
        pipeline = EnrichmentPipeline(backend=MockEnrichmentBackend())
        result = pipeline.enrich([], "test")
        assert result == []

    def test_stats_tracking(self):
        backend = MockEnrichmentBackend()
        pipeline = EnrichmentPipeline(backend=backend)
        rules = _make_rules(7)

        pipeline.enrich(rules, "test")

        assert pipeline.stats.total == 7
        assert pipeline.stats.succeeded == 7
        assert pipeline.stats.failed == 0

    def test_stats_reset_on_each_enrich(self):
        backend = MockEnrichmentBackend()
        pipeline = EnrichmentPipeline(backend=backend)

        pipeline.enrich(_make_rules(3), "test")
        assert pipeline.stats.total == 3

        pipeline.enrich(_make_rules(5), "test")
        assert pipeline.stats.total == 5  # Reset, not accumulated

    def test_enriched_rules_preserve_original(self):
        backend = MockEnrichmentBackend()
        pipeline = EnrichmentPipeline(backend=backend)
        rules = _make_rules(2)

        result = pipeline.enrich(rules, "test")

        assert result[0].rule is rules[0]
        assert result[1].rule is rules[1]


# -------------------------------------------------------------------------
# Anthropic Backend (import test only — no API calls)
# -------------------------------------------------------------------------


class TestAnthropicBackendImport:
    def test_parse_json_strips_fences(self):
        """Test JSON parsing with markdown code fences."""
        # Only test if anthropic is installed
        try:
            from qortex.enrichment.anthropic import AnthropicEnrichmentBackend
        except ImportError:
            pytest.skip("anthropic not installed")

        # Can't instantiate without API key, so test the parser directly
        # by monkey-patching
        backend = object.__new__(AnthropicEnrichmentBackend)
        result = backend._parse_json('```json\n[{"context": "test"}]\n```')
        assert result == [{"context": "test"}]

    def test_parse_json_plain(self):
        try:
            from qortex.enrichment.anthropic import AnthropicEnrichmentBackend
        except ImportError:
            pytest.skip("anthropic not installed")

        backend = object.__new__(AnthropicEnrichmentBackend)
        result = backend._parse_json('[{"context": "test"}]')
        assert result == [{"context": "test"}]
