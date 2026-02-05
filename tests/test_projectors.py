"""Exhaustive tests for projection protocols, models, orchestrator, and registry."""

import pytest

from qortex.core.models import RelationType, Rule
from qortex.projectors.base import Enricher, ProjectionSource, ProjectionTarget
from qortex.projectors.models import EnrichedRule, ProjectionFilter, RuleEnrichment
from qortex.projectors.projection import Projection
from qortex.projectors import registry


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------


def _make_rule(id: str = "r1", text: str = "Test rule", domain: str = "test") -> Rule:
    return Rule(
        id=id,
        text=text,
        domain=domain,
        derivation="explicit",
        source_concepts=["c1"],
        confidence=0.9,
    )


def _make_enrichment() -> RuleEnrichment:
    return RuleEnrichment(
        context="When testing",
        antipattern="Not testing",
        rationale="Testing prevents bugs",
        tags=["testing"],
    )


# -------------------------------------------------------------------------
# Stub implementations for testing
# -------------------------------------------------------------------------


class StubSource:
    def __init__(self, rules: list[Rule] | None = None):
        self._rules = rules or []

    def derive(self, domains=None, filters=None):
        return self._rules


class StubEnricher:
    def enrich(self, rules):
        return [
            EnrichedRule(rule=r, enrichment=_make_enrichment())
            for r in rules
        ]


class StubTarget:
    def serialize(self, rules):
        return [{"id": r.id if isinstance(r, Rule) else r.rule.id} for r in rules]


# -------------------------------------------------------------------------
# Model Tests
# -------------------------------------------------------------------------


class TestProjectionFilter:
    def test_defaults(self):
        f = ProjectionFilter()
        assert f.domains is None
        assert f.categories is None
        assert f.min_confidence == 0.0
        assert f.derivation == "all"
        assert f.relation_types is None

    def test_custom_values(self):
        f = ProjectionFilter(
            domains=["a"],
            categories=["arch"],
            min_confidence=0.5,
            derivation="explicit",
            relation_types=[RelationType.REQUIRES],
        )
        assert f.domains == ["a"]
        assert f.min_confidence == 0.5


class TestRuleEnrichment:
    def test_creation(self):
        e = _make_enrichment()
        assert e.context == "When testing"
        assert e.enrichment_version == 1
        assert e.enrichment_source is None

    def test_defaults(self):
        e = RuleEnrichment(
            context="c", antipattern="a", rationale="r", tags=["t"]
        )
        assert e.enrichment_version == 1
        assert e.source_contexts == []


class TestEnrichedRule:
    def test_creation(self):
        rule = _make_rule()
        enrichment = _make_enrichment()
        er = EnrichedRule(rule=rule, enrichment=enrichment)
        assert er.rule is rule
        assert er.enrichment is enrichment

    def test_delegates_id(self):
        er = EnrichedRule(rule=_make_rule(id="test_id"))
        assert er.id == "test_id"

    def test_delegates_text(self):
        er = EnrichedRule(rule=_make_rule(text="hello"))
        assert er.text == "hello"

    def test_delegates_domain(self):
        er = EnrichedRule(rule=_make_rule(domain="fp"))
        assert er.domain == "fp"

    def test_delegates_confidence(self):
        er = EnrichedRule(rule=_make_rule())
        assert er.confidence == 0.9

    def test_no_enrichment(self):
        er = EnrichedRule(rule=_make_rule())
        assert er.enrichment is None


# -------------------------------------------------------------------------
# Protocol Compliance
# -------------------------------------------------------------------------


class TestProtocols:
    def test_stub_source_is_projection_source(self):
        assert isinstance(StubSource(), ProjectionSource)

    def test_stub_enricher_is_enricher(self):
        assert isinstance(StubEnricher(), Enricher)

    def test_stub_target_is_projection_target(self):
        assert isinstance(StubTarget(), ProjectionTarget)


# -------------------------------------------------------------------------
# Projection Orchestrator
# -------------------------------------------------------------------------


class TestProjection:
    def test_project_without_enricher(self):
        rules = [_make_rule("r1"), _make_rule("r2")]
        p = Projection(
            source=StubSource(rules),
            target=StubTarget(),
        )
        result = p.project()
        assert len(result) == 2
        assert result[0]["id"] == "r1"

    def test_project_with_enricher(self):
        rules = [_make_rule("r1")]
        p = Projection(
            source=StubSource(rules),
            enricher=StubEnricher(),
            target=StubTarget(),
        )
        result = p.project()
        assert len(result) == 1

    def test_project_with_domain_filter(self):
        rules = [_make_rule("r1")]
        source = StubSource(rules)
        p = Projection(source=source, target=StubTarget())
        # Domain is passed through to source.derive() — stub ignores it
        result = p.project(domains=["error_handling"])
        assert len(result) == 1

    def test_project_empty_rules(self):
        p = Projection(source=StubSource([]), target=StubTarget())
        result = p.project()
        assert result == []


# -------------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------------


class TestRegistry:
    def setup_method(self):
        # Clean registry state
        registry._sources.clear()
        registry._enrichers.clear()
        registry._targets.clear()

    def test_register_and_get_source(self):
        registry.register_source("stub", StubSource)
        src = registry.get_source("stub")
        assert isinstance(src, StubSource)

    def test_register_and_get_enricher(self):
        registry.register_enricher("stub", StubEnricher)
        e = registry.get_enricher("stub")
        assert isinstance(e, StubEnricher)

    def test_register_and_get_target(self):
        registry.register_target("stub", StubTarget)
        t = registry.get_target("stub")
        assert isinstance(t, StubTarget)

    def test_get_unknown_source_raises(self):
        with pytest.raises(KeyError, match="Unknown projection source"):
            registry.get_source("nonexistent")

    def test_get_unknown_enricher_raises(self):
        with pytest.raises(KeyError, match="Unknown enricher"):
            registry.get_enricher("nonexistent")

    def test_get_unknown_target_raises(self):
        with pytest.raises(KeyError, match="Unknown projection target"):
            registry.get_target("nonexistent")

    def test_available_sources(self):
        registry.register_source("a", StubSource)
        registry.register_source("b", StubSource)
        assert set(registry.available_sources()) == {"a", "b"}

    def test_available_enrichers(self):
        registry.register_enricher("x", StubEnricher)
        assert registry.available_enrichers() == ["x"]

    def test_available_targets(self):
        registry.register_target("y", StubTarget)
        assert registry.available_targets() == ["y"]
