"""Tests for CausalRuleProjector.

Protocol compliance + integration with Projection pipeline.
"""

from qortex.causal.projector import CausalRuleProjector
from qortex.projectors.base import ProjectionSource
from qortex.projectors.models import ProjectionFilter


class TestProtocolCompliance:
    def test_is_projection_source(self, backend_with_graph):
        projector = CausalRuleProjector(backend=backend_with_graph)
        assert isinstance(projector, ProjectionSource)

    def test_derive_signature(self, backend_with_graph):
        projector = CausalRuleProjector(backend=backend_with_graph)
        rules = projector.derive(domains=["test"])
        assert isinstance(rules, list)


class TestDerive:
    def test_derives_causal_rules(self, backend_with_graph):
        projector = CausalRuleProjector(backend=backend_with_graph)
        rules = projector.derive(domains=["test"])
        assert len(rules) > 0
        for rule in rules:
            assert rule.derivation == "causal"

    def test_rules_have_correct_domain(self, backend_with_graph):
        projector = CausalRuleProjector(backend=backend_with_graph)
        rules = projector.derive(domains=["test"])
        for rule in rules:
            assert rule.domain == "test"

    def test_rules_have_source_concepts(self, backend_with_graph):
        projector = CausalRuleProjector(backend=backend_with_graph)
        rules = projector.derive(domains=["test"])
        for rule in rules:
            assert len(rule.source_concepts) >= 2

    def test_skip_non_causal_filter(self, backend_with_graph):
        projector = CausalRuleProjector(backend=backend_with_graph)
        filt = ProjectionFilter(derivation="explicit")
        rules = projector.derive(domains=["test"], filters=filt)
        assert rules == []

    def test_causal_filter_works(self, backend_with_graph):
        projector = CausalRuleProjector(backend=backend_with_graph)
        filt = ProjectionFilter(derivation="causal")
        rules = projector.derive(domains=["test"], filters=filt)
        assert len(rules) > 0

    def test_all_filter_includes_causal(self, backend_with_graph):
        projector = CausalRuleProjector(backend=backend_with_graph)
        filt = ProjectionFilter(derivation="all")
        rules = projector.derive(domains=["test"], filters=filt)
        assert len(rules) > 0

    def test_deduplicates_pairs(self, backend_with_graph):
        projector = CausalRuleProjector(backend=backend_with_graph)
        rules = projector.derive(domains=["test"])
        # Check no duplicate (x, y) pairs
        pairs = set()
        for rule in rules:
            pair = tuple(sorted(rule.source_concepts[:2]))
            assert pair not in pairs, f"Duplicate pair: {pair}"
            pairs.add(pair)

    def test_min_confidence_filter(self, backend_with_graph):
        projector = CausalRuleProjector(backend=backend_with_graph)
        filt = ProjectionFilter(derivation="causal", min_confidence=0.99)
        rules = projector.derive(domains=["test"], filters=filt)
        for rule in rules:
            assert rule.confidence >= 0.99

    def test_empty_domain_returns_empty(self, backend_with_graph):
        projector = CausalRuleProjector(backend=backend_with_graph)
        rules = projector.derive(domains=["nonexistent"])
        assert rules == []


class TestRuleContent:
    def test_independence_rule_text(self, backend_with_graph):
        projector = CausalRuleProjector(backend=backend_with_graph)
        rules = projector.derive(domains=["test"])
        independence_rules = [r for r in rules if r.category == "causal_independence"]
        assert len(independence_rules) > 0
        for rule in independence_rules:
            assert "independent" in rule.text.lower()

    def test_rule_metadata_has_method(self, backend_with_graph):
        projector = CausalRuleProjector(backend=backend_with_graph)
        rules = projector.derive(domains=["test"])
        for rule in rules:
            assert "method" in rule.metadata
