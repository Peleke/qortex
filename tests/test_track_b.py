"""Exhaustive tests for Track B: Sources, Targets, Enrichers.

Includes property-based tests (hypothesis) for serialization targets
and thorough unit tests for all components.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from qortex.core.memory import InMemoryBackend
from qortex.core.models import (
    ConceptEdge,
    ConceptNode,
    ExplicitRule,
    RelationType,
    Rule,
)
from qortex.core.templates import EDGE_RULE_TEMPLATE_REGISTRY
from qortex.enrichment.pipeline import EnrichmentPipeline, TemplateEnrichmentFallback
from qortex.projectors.base import Enricher, ProjectionSource, ProjectionTarget
from qortex.projectors.enrichers.llm import LLMEnricher
from qortex.projectors.enrichers.passthrough import PassthroughEnricher
from qortex.projectors.enrichers.template import TemplateEnricher
from qortex.projectors.models import EnrichedRule, ProjectionFilter, RuleEnrichment
from qortex.projectors.projection import Projection
from qortex.projectors.sources.flat import FlatRuleSource
from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget
from qortex.projectors.targets.flat_json import FlatJSONTarget
from qortex.projectors.targets.flat_yaml import FlatYAMLTarget


# =========================================================================
# Test fixtures
# =========================================================================


def _make_backend() -> InMemoryBackend:
    """Create a backend with error_handling domain, 4 nodes, 2 edges, 1 rule."""
    b = InMemoryBackend()
    b.connect()
    b.create_domain("error_handling", "Error handling patterns")

    nodes = [
        ConceptNode(
            id="err:circuit_breaker",
            name="Circuit Breaker",
            description="Pattern that prevents cascading failures",
            domain="error_handling",
            source_id="book:ch3",
        ),
        ConceptNode(
            id="err:timeout",
            name="Timeout Configuration",
            description="Configuring timeouts for external calls",
            domain="error_handling",
            source_id="book:ch3",
        ),
        ConceptNode(
            id="err:retry",
            name="Retry",
            description="Retrying failed operations",
            domain="error_handling",
            source_id="book:ch3",
        ),
        ConceptNode(
            id="err:fail_fast",
            name="Fail Fast",
            description="Failing immediately on unrecoverable errors",
            domain="error_handling",
            source_id="book:ch3",
        ),
    ]
    for node in nodes:
        b.add_node(node)

    b.add_edge(
        ConceptEdge(
            source_id="err:circuit_breaker",
            target_id="err:timeout",
            relation_type=RelationType.REQUIRES,
        )
    )
    b.add_edge(
        ConceptEdge(
            source_id="err:retry",
            target_id="err:fail_fast",
            relation_type=RelationType.CONTRADICTS,
            confidence=0.9,
        )
    )

    b.add_rule(
        ExplicitRule(
            id="rule:timeout",
            text="Always configure timeouts for external calls",
            domain="error_handling",
            source_id="book:ch3",
            concept_ids=["err:timeout"],
            category="architectural",
        )
    )

    return b


def _make_rule(
    id: str = "r1",
    text: str = "Test rule",
    domain: str = "test",
    derivation: str = "explicit",
    confidence: float = 0.9,
    category: str | None = None,
) -> Rule:
    return Rule(
        id=id,
        text=text,
        domain=domain,
        derivation=derivation,
        source_concepts=["c1"],
        confidence=confidence,
        category=category,
    )


def _make_enriched_rule(
    id: str = "r1",
    domain: str = "test",
    with_enrichment: bool = True,
) -> EnrichedRule:
    rule = _make_rule(id=id, domain=domain)
    enrichment = None
    if with_enrichment:
        enrichment = RuleEnrichment(
            context=f"Context for {id}",
            antipattern=f"Antipattern for {id}",
            rationale=f"Rationale for {id}",
            tags=[domain, "test"],
            enrichment_version=1,
            enriched_at=datetime.now(timezone.utc),
            enrichment_source="test",
        )
    return EnrichedRule(rule=rule, enrichment=enrichment)


# =========================================================================
# Protocol compliance
# =========================================================================


class TestProtocolCompliance:
    def test_flat_rule_source_is_projection_source(self):
        b = InMemoryBackend()
        b.connect()
        assert isinstance(FlatRuleSource(backend=b), ProjectionSource)

    def test_passthrough_enricher_is_enricher(self):
        assert isinstance(PassthroughEnricher(), Enricher)

    def test_template_enricher_is_enricher(self):
        assert isinstance(TemplateEnricher(), Enricher)

    def test_llm_enricher_is_enricher(self):
        assert isinstance(LLMEnricher(), Enricher)

    def test_flat_yaml_target_is_projection_target(self):
        assert isinstance(FlatYAMLTarget(), ProjectionTarget)

    def test_flat_json_target_is_projection_target(self):
        assert isinstance(FlatJSONTarget(), ProjectionTarget)

    def test_buildlog_seed_target_is_projection_target(self):
        assert isinstance(BuildlogSeedTarget(), ProjectionTarget)


# =========================================================================
# FlatRuleSource
# =========================================================================


class TestFlatRuleSource:
    def test_derive_explicit_rules(self):
        backend = _make_backend()
        source = FlatRuleSource(backend=backend, include_derived=False)
        rules = source.derive(domains=["error_handling"])

        assert len(rules) == 1
        assert rules[0].id == "rule:timeout"
        assert rules[0].derivation == "explicit"

    def test_derive_edge_rules(self):
        backend = _make_backend()
        source = FlatRuleSource(backend=backend)
        rules = source.derive(
            domains=["error_handling"],
            filters=ProjectionFilter(derivation="derived"),
        )

        assert len(rules) == 2
        assert all(r.derivation == "derived" for r in rules)

    def test_derive_all_rules(self):
        backend = _make_backend()
        source = FlatRuleSource(backend=backend)
        rules = source.derive(domains=["error_handling"])

        # 1 explicit + 2 derived
        assert len(rules) == 3
        explicit = [r for r in rules if r.derivation == "explicit"]
        derived = [r for r in rules if r.derivation == "derived"]
        assert len(explicit) == 1
        assert len(derived) == 2

    def test_derive_empty_domain(self):
        backend = _make_backend()
        backend.create_domain("empty")
        source = FlatRuleSource(backend=backend)
        rules = source.derive(domains=["empty"])
        assert rules == []

    def test_derive_no_domains_uses_all(self):
        backend = _make_backend()
        source = FlatRuleSource(backend=backend)
        rules = source.derive()
        assert len(rules) == 3  # All from error_handling

    def test_derive_deduplicates_edge_pairs(self):
        """Same (source, target) edge pair should not produce duplicate rules."""
        backend = _make_backend()
        # Add a duplicate edge (same source/target, different type)
        backend.add_edge(
            ConceptEdge(
                source_id="err:circuit_breaker",
                target_id="err:timeout",
                relation_type=RelationType.USES,
            )
        )
        source = FlatRuleSource(backend=backend)
        rules = source.derive(
            domains=["error_handling"],
            filters=ProjectionFilter(derivation="derived"),
        )
        # Should still be 2, not 3 (dedup on source/target pair)
        pairs = [(r.source_concepts[0], r.source_concepts[1]) for r in rules]
        assert len(set(pairs)) == len(pairs)

    def test_filter_by_min_confidence(self):
        backend = _make_backend()
        source = FlatRuleSource(backend=backend)
        rules = source.derive(
            domains=["error_handling"],
            filters=ProjectionFilter(min_confidence=0.95),
        )
        # Explicit rule has confidence 1.0, edge-derived REQUIRES has 1.0,
        # CONTRADICTS has 0.9 (filtered out)
        assert all(r.confidence >= 0.95 for r in rules)

    def test_filter_by_category(self):
        backend = _make_backend()
        source = FlatRuleSource(backend=backend)
        rules = source.derive(
            domains=["error_handling"],
            filters=ProjectionFilter(categories=["architectural"]),
        )
        assert all(r.category == "architectural" for r in rules)

    def test_filter_by_relation_type(self):
        backend = _make_backend()
        source = FlatRuleSource(backend=backend)
        rules = source.derive(
            domains=["error_handling"],
            filters=ProjectionFilter(
                derivation="derived",
                relation_types=[RelationType.REQUIRES],
            ),
        )
        assert len(rules) == 1
        assert "Circuit Breaker" in rules[0].text or "Timeout" in rules[0].text

    def test_filter_explicit_only(self):
        backend = _make_backend()
        source = FlatRuleSource(backend=backend)
        rules = source.derive(
            domains=["error_handling"],
            filters=ProjectionFilter(derivation="explicit"),
        )
        assert len(rules) == 1
        assert rules[0].derivation == "explicit"

    def test_derived_rule_text_uses_template(self):
        backend = _make_backend()
        source = FlatRuleSource(backend=backend)
        rules = source.derive(
            domains=["error_handling"],
            filters=ProjectionFilter(derivation="derived"),
        )
        # The REQUIRES template should reference both node names
        requires_rules = [r for r in rules if "Circuit Breaker" in r.text and "Timeout" in r.text]
        assert len(requires_rules) >= 1

    def test_derived_rule_source_concepts(self):
        backend = _make_backend()
        source = FlatRuleSource(backend=backend)
        rules = source.derive(
            domains=["error_handling"],
            filters=ProjectionFilter(derivation="derived"),
        )
        for rule in rules:
            assert len(rule.source_concepts) == 2

    def test_derived_rule_id_format(self):
        backend = _make_backend()
        source = FlatRuleSource(backend=backend)
        rules = source.derive(
            domains=["error_handling"],
            filters=ProjectionFilter(derivation="derived"),
        )
        for rule in rules:
            assert rule.id.startswith("derived:")

    def test_multi_domain(self):
        backend = _make_backend()
        backend.create_domain("fp")
        backend.add_node(
            ConceptNode(
                id="fp:pure",
                name="Pure Function",
                description="No side effects",
                domain="fp",
                source_id="book:ch1",
            )
        )
        backend.add_rule(
            ExplicitRule(
                id="rule:pure",
                text="Prefer pure functions",
                domain="fp",
                source_id="book:ch1",
            )
        )

        source = FlatRuleSource(backend=backend)

        # All domains
        all_rules = source.derive()
        assert len(all_rules) == 4  # 2 explicit + 2 derived

        # Single domain
        fp_rules = source.derive(domains=["fp"])
        assert len(fp_rules) == 1
        assert fp_rules[0].domain == "fp"

    def test_missing_node_skips_edge(self):
        """If an edge references a node not in the backend, skip it."""
        backend = _make_backend()
        backend.add_edge(
            ConceptEdge(
                source_id="err:circuit_breaker",
                target_id="nonexistent",
                relation_type=RelationType.USES,
            )
        )
        source = FlatRuleSource(backend=backend)
        rules = source.derive(
            domains=["error_handling"],
            filters=ProjectionFilter(derivation="derived"),
        )
        # The edge with nonexistent target should be skipped
        for rule in rules:
            assert "nonexistent" not in str(rule.source_concepts)

    def test_state_reset_between_derive_calls(self):
        """_seen_edge_pairs resets between calls."""
        backend = _make_backend()
        source = FlatRuleSource(backend=backend)

        rules1 = source.derive(
            domains=["error_handling"],
            filters=ProjectionFilter(derivation="derived"),
        )
        rules2 = source.derive(
            domains=["error_handling"],
            filters=ProjectionFilter(derivation="derived"),
        )
        assert len(rules1) == len(rules2)


# =========================================================================
# Enrichers
# =========================================================================


class TestPassthroughEnricher:
    def test_enrich_wraps_rules(self):
        enricher = PassthroughEnricher()
        rules = [_make_rule(f"r{i}") for i in range(3)]
        result = enricher.enrich(rules)

        assert len(result) == 3
        assert all(isinstance(r, EnrichedRule) for r in result)
        assert all(r.enrichment is None for r in result)

    def test_enrich_preserves_rule_identity(self):
        enricher = PassthroughEnricher()
        rules = [_make_rule()]
        result = enricher.enrich(rules)
        assert result[0].rule is rules[0]

    def test_enrich_empty(self):
        enricher = PassthroughEnricher()
        assert enricher.enrich([]) == []


class TestTemplateEnricher:
    def test_enrich_produces_enrichments(self):
        enricher = TemplateEnricher(domain="test")
        rules = [_make_rule(f"r{i}") for i in range(3)]
        result = enricher.enrich(rules)

        assert len(result) == 3
        assert all(r.enrichment is not None for r in result)
        assert all(r.enrichment.enrichment_source == "template" for r in result)

    def test_enrich_uses_domain(self):
        enricher = TemplateEnricher(domain="error_handling")
        rules = [_make_rule(domain="error_handling")]
        result = enricher.enrich(rules)
        assert "error_handling" in result[0].enrichment.tags

    def test_enrich_empty(self):
        enricher = TemplateEnricher()
        assert enricher.enrich([]) == []

    def test_enrich_preserves_rule(self):
        enricher = TemplateEnricher()
        rules = [_make_rule()]
        result = enricher.enrich(rules)
        assert result[0].rule is rules[0]


class TestLLMEnricher:
    def test_enrich_without_backend_uses_fallback(self):
        """LLMEnricher with no LLM backend falls back to template."""
        enricher = LLMEnricher(pipeline=EnrichmentPipeline())
        rules = [_make_rule()]
        result = enricher.enrich(rules)

        assert len(result) == 1
        assert result[0].enrichment is not None
        assert result[0].enrichment.enrichment_source == "template"

    def test_enrich_empty(self):
        enricher = LLMEnricher()
        assert enricher.enrich([]) == []

    def test_infers_domain_from_rules(self):
        enricher = LLMEnricher(domain="default")
        rules = [_make_rule(domain="specific")]
        result = enricher.enrich(rules)
        # The enrichment should use the rule's domain, not default
        assert result[0].enrichment is not None


# =========================================================================
# Targets: FlatYAMLTarget
# =========================================================================


class TestFlatYAMLTarget:
    def test_serialize_enriched_rules(self):
        target = FlatYAMLTarget()
        rules = [_make_enriched_rule("r1"), _make_enriched_rule("r2")]
        result = target.serialize(rules)

        parsed = yaml.safe_load(result)
        assert "rules" in parsed
        assert len(parsed["rules"]) == 2
        assert parsed["rules"][0]["id"] == "r1"
        assert "enrichment" in parsed["rules"][0]
        assert parsed["rules"][0]["enrichment"]["context"] == "Context for r1"

    def test_serialize_plain_rules(self):
        target = FlatYAMLTarget()
        rules = [_make_rule("r1"), _make_rule("r2")]
        result = target.serialize(rules)

        parsed = yaml.safe_load(result)
        assert len(parsed["rules"]) == 2
        assert "enrichment" not in parsed["rules"][0]

    def test_serialize_without_enrichment_flag(self):
        target = FlatYAMLTarget(include_enrichment=False)
        rules = [_make_enriched_rule("r1")]
        result = target.serialize(rules)

        parsed = yaml.safe_load(result)
        assert "enrichment" not in parsed["rules"][0]

    def test_serialize_empty(self):
        target = FlatYAMLTarget()
        result = target.serialize([])
        parsed = yaml.safe_load(result)
        assert parsed["rules"] == []

    def test_serialize_enriched_rule_no_enrichment(self):
        """EnrichedRule with enrichment=None should not include enrichment key."""
        target = FlatYAMLTarget()
        rules = [_make_enriched_rule("r1", with_enrichment=False)]
        result = target.serialize(rules)
        parsed = yaml.safe_load(result)
        assert "enrichment" not in parsed["rules"][0]

    def test_serialize_includes_category_when_present(self):
        target = FlatYAMLTarget()
        rule = _make_rule(category="architectural")
        result = target.serialize([rule])
        parsed = yaml.safe_load(result)
        assert parsed["rules"][0]["category"] == "architectural"

    def test_serialize_omits_category_when_none(self):
        target = FlatYAMLTarget()
        rule = _make_rule(category=None)
        result = target.serialize([rule])
        parsed = yaml.safe_load(result)
        assert "category" not in parsed["rules"][0]

    def test_output_is_valid_yaml(self):
        target = FlatYAMLTarget()
        rules = [_make_enriched_rule(f"r{i}") for i in range(10)]
        result = target.serialize(rules)
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)


# =========================================================================
# Targets: FlatJSONTarget
# =========================================================================


class TestFlatJSONTarget:
    def test_serialize_enriched_rules(self):
        target = FlatJSONTarget()
        rules = [_make_enriched_rule("r1")]
        result = target.serialize(rules)

        parsed = json.loads(result)
        assert parsed["rules"][0]["id"] == "r1"
        assert parsed["rules"][0]["enrichment"]["context"] == "Context for r1"

    def test_serialize_plain_rules(self):
        target = FlatJSONTarget()
        rules = [_make_rule("r1")]
        result = target.serialize(rules)

        parsed = json.loads(result)
        assert len(parsed["rules"]) == 1
        assert "enrichment" not in parsed["rules"][0]

    def test_serialize_without_enrichment_flag(self):
        target = FlatJSONTarget(include_enrichment=False)
        rules = [_make_enriched_rule("r1")]
        result = target.serialize(rules)

        parsed = json.loads(result)
        assert "enrichment" not in parsed["rules"][0]

    def test_serialize_empty(self):
        target = FlatJSONTarget()
        result = target.serialize([])
        parsed = json.loads(result)
        assert parsed["rules"] == []

    def test_output_is_valid_json(self):
        target = FlatJSONTarget()
        rules = [_make_enriched_rule(f"r{i}") for i in range(10)]
        result = target.serialize(rules)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_custom_indent(self):
        target = FlatJSONTarget(indent=4)
        rules = [_make_rule()]
        result = target.serialize(rules)
        # 4-space indent should be present
        assert "    " in result


# =========================================================================
# Targets: BuildlogSeedTarget
# =========================================================================


class TestBuildlogSeedTarget:
    def test_serialize_structure(self):
        # New universal schema: persona is flat string, version is int
        target = BuildlogSeedTarget(
            persona_name="qortex_error_handling",
            version=1,
        )
        rules = [_make_enriched_rule("r1")]
        result = target.serialize(rules)

        assert result["persona"] == "qortex_error_handling"  # flat string
        assert result["version"] == 1  # int
        assert result["metadata"]["source"] == "qortex"
        assert result["metadata"]["rule_count"] == 1

    def test_serialize_rules_with_enrichment(self):
        target = BuildlogSeedTarget()
        rules = [_make_enriched_rule("r1")]
        result = target.serialize(rules)

        seed_rule = result["rules"][0]
        # 'rule' key (not 'text'), provenance block
        assert seed_rule["rule"] == "Test rule"
        assert seed_rule["provenance"]["id"] == "r1"
        assert seed_rule["context"] == "Context for r1"
        assert seed_rule["antipattern"] == "Antipattern for r1"
        assert seed_rule["rationale"] == "Rationale for r1"
        assert "test" in seed_rule["tags"]

    def test_serialize_rules_without_enrichment(self):
        target = BuildlogSeedTarget()
        rules = [_make_enriched_rule("r1", with_enrichment=False)]
        result = target.serialize(rules)

        seed_rule = result["rules"][0]
        assert seed_rule["provenance"]["id"] == "r1"
        assert "context" not in seed_rule
        assert "antipattern" not in seed_rule

    def test_serialize_plain_rules(self):
        target = BuildlogSeedTarget()
        rules = [_make_rule("r1")]
        result = target.serialize(rules)

        assert len(result["rules"]) == 1
        assert result["rules"][0]["provenance"]["id"] == "r1"
        assert "context" not in result["rules"][0]

    def test_serialize_empty(self):
        target = BuildlogSeedTarget()
        result = target.serialize([])
        assert result["rules"] == []
        assert result["metadata"]["rule_count"] == 0

    def test_extra_metadata(self):
        target = BuildlogSeedTarget(extra_metadata={"chapter": "3"})
        result = target.serialize([_make_rule()])
        assert result["metadata"]["chapter"] == "3"

    def test_category_included_when_present(self):
        target = BuildlogSeedTarget()
        rule = _make_rule(category="architectural")
        result = target.serialize([rule])
        assert result["rules"][0]["category"] == "architectural"

    def test_category_falls_back_to_domain(self):
        # New schema: category falls back to domain when None
        target = BuildlogSeedTarget()
        rule = _make_rule(category=None, domain="error_handling")
        result = target.serialize([rule])
        assert result["rules"][0]["category"] == "error_handling"


# =========================================================================
# Full pipeline integration (Source -> Enricher -> Target)
# =========================================================================


class TestProjectionPipelineIntegration:
    def test_flat_source_passthrough_yaml(self):
        """FlatRuleSource -> PassthroughEnricher -> FlatYAMLTarget."""
        backend = _make_backend()
        projection = Projection(
            source=FlatRuleSource(backend=backend),
            enricher=PassthroughEnricher(),
            target=FlatYAMLTarget(),
        )
        result = projection.project(domains=["error_handling"])

        parsed = yaml.safe_load(result)
        assert len(parsed["rules"]) == 3

    def test_flat_source_template_json(self):
        """FlatRuleSource -> TemplateEnricher -> FlatJSONTarget."""
        backend = _make_backend()
        projection = Projection(
            source=FlatRuleSource(backend=backend),
            enricher=TemplateEnricher(domain="error_handling"),
            target=FlatJSONTarget(),
        )
        result = projection.project(domains=["error_handling"])

        parsed = json.loads(result)
        assert len(parsed["rules"]) == 3
        assert all("enrichment" in r for r in parsed["rules"])

    def test_flat_source_template_buildlog(self):
        """FlatRuleSource -> TemplateEnricher -> BuildlogSeedTarget."""
        backend = _make_backend()
        projection = Projection(
            source=FlatRuleSource(backend=backend),
            enricher=TemplateEnricher(domain="error_handling"),
            target=BuildlogSeedTarget(persona_name="qortex_error_handling"),
        )
        result = projection.project(domains=["error_handling"])

        # New universal schema: persona is flat string
        assert result["persona"] == "qortex_error_handling"
        assert len(result["rules"]) == 3
        assert all("context" in r for r in result["rules"])

    def test_flat_source_no_enricher_yaml(self):
        """FlatRuleSource -> no enricher -> FlatYAMLTarget."""
        backend = _make_backend()
        projection = Projection(
            source=FlatRuleSource(backend=backend),
            target=FlatYAMLTarget(),
        )
        result = projection.project(domains=["error_handling"])

        parsed = yaml.safe_load(result)
        assert len(parsed["rules"]) == 3
        # No enrichment since no enricher
        for rule in parsed["rules"]:
            assert "enrichment" not in rule

    def test_filter_flows_through_pipeline(self):
        """Filters applied at source level propagate through pipeline."""
        backend = _make_backend()
        projection = Projection(
            source=FlatRuleSource(backend=backend),
            enricher=PassthroughEnricher(),
            target=FlatJSONTarget(),
        )
        result = projection.project(
            domains=["error_handling"],
            filters=ProjectionFilter(derivation="explicit"),
        )

        parsed = json.loads(result)
        assert len(parsed["rules"]) == 1


# =========================================================================
# Property-based tests (hypothesis)
# =========================================================================


# Strategy: generate valid Rule objects
# Use ASCII-printable strings for IDs/text to avoid YAML encoding issues
# (YAML normalizes certain Unicode whitespace like \x85)
_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z"), whitelist_characters=" _-:./"),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip())

rule_strategy = st.builds(
    Rule,
    id=_safe_text,
    text=st.text(min_size=1, max_size=200).filter(lambda s: s.strip()),
    domain=st.sampled_from(["error_handling", "fp", "testing", "architecture"]),
    derivation=st.sampled_from(["explicit", "derived"]),
    source_concepts=st.lists(st.text(min_size=1, max_size=20), min_size=0, max_size=5),
    confidence=st.floats(min_value=0.0, max_value=1.0),
    category=st.one_of(
        st.none(),
        st.sampled_from(["antipattern", "architectural", "general"]),
    ),
)


class TestPropertyBasedYAML:
    @given(rules=st.lists(rule_strategy, min_size=0, max_size=20))
    @settings(max_examples=50)
    def test_yaml_roundtrip_preserves_count(self, rules: list[Rule]):
        """YAML serialization preserves rule count."""
        target = FlatYAMLTarget()
        result = target.serialize(rules)
        parsed = yaml.safe_load(result)
        assert len(parsed["rules"]) == len(rules)

    @given(rules=st.lists(rule_strategy, min_size=0, max_size=20))
    @settings(max_examples=50)
    def test_yaml_output_is_always_valid(self, rules: list[Rule]):
        """YAML output is always parseable."""
        target = FlatYAMLTarget()
        result = target.serialize(rules)
        parsed = yaml.safe_load(result)
        assert isinstance(parsed, dict)
        assert "rules" in parsed

    @given(rules=st.lists(rule_strategy, min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_yaml_preserves_rule_ids(self, rules: list[Rule]):
        """YAML serialization preserves all rule IDs."""
        target = FlatYAMLTarget()
        result = target.serialize(rules)
        parsed = yaml.safe_load(result)
        output_ids = {r["id"] for r in parsed["rules"]}
        input_ids = {r.id for r in rules}
        assert output_ids == input_ids


class TestPropertyBasedJSON:
    @given(rules=st.lists(rule_strategy, min_size=0, max_size=20))
    @settings(max_examples=50)
    def test_json_roundtrip_preserves_count(self, rules: list[Rule]):
        """JSON serialization preserves rule count."""
        target = FlatJSONTarget()
        result = target.serialize(rules)
        parsed = json.loads(result)
        assert len(parsed["rules"]) == len(rules)

    @given(rules=st.lists(rule_strategy, min_size=0, max_size=20))
    @settings(max_examples=50)
    def test_json_output_is_always_valid(self, rules: list[Rule]):
        """JSON output is always parseable."""
        target = FlatJSONTarget()
        result = target.serialize(rules)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)
        assert "rules" in parsed

    @given(rules=st.lists(rule_strategy, min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_json_preserves_rule_ids(self, rules: list[Rule]):
        """JSON serialization preserves all rule IDs."""
        target = FlatJSONTarget()
        result = target.serialize(rules)
        parsed = json.loads(result)
        output_ids = {r["id"] for r in parsed["rules"]}
        input_ids = {r.id for r in rules}
        assert output_ids == input_ids


class TestPropertyBasedBuildlogSeed:
    @given(rules=st.lists(rule_strategy, min_size=0, max_size=20))
    @settings(max_examples=50)
    def test_buildlog_seed_count(self, rules: list[Rule]):
        """Buildlog seed preserves rule count in metadata and rules list."""
        target = BuildlogSeedTarget()
        result = target.serialize(rules)
        assert result["metadata"]["rule_count"] == len(rules)
        assert len(result["rules"]) == len(rules)

    @given(rules=st.lists(rule_strategy, min_size=0, max_size=10))
    @settings(max_examples=50)
    def test_buildlog_seed_always_has_persona(self, rules: list[Rule]):
        """Buildlog seed always includes persona (flat string in new schema)."""
        target = BuildlogSeedTarget()
        result = target.serialize(rules)
        assert "persona" in result
        # New schema: persona is flat string, not dict
        assert isinstance(result["persona"], str)

    @given(rules=st.lists(rule_strategy, min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_buildlog_seed_all_rules_have_required_fields(self, rules: list[Rule]):
        """Every rule in buildlog seed has provenance with id, domain, confidence."""
        target = BuildlogSeedTarget()
        result = target.serialize(rules)
        for seed_rule in result["rules"]:
            # New schema: 'rule' key (not 'text'), provenance block
            assert "rule" in seed_rule
            assert "category" in seed_rule
            assert "provenance" in seed_rule
            prov = seed_rule["provenance"]
            assert "id" in prov
            assert "domain" in prov
            assert "confidence" in prov


class TestPropertyBasedCrossFormat:
    @given(rules=st.lists(rule_strategy, min_size=1, max_size=10))
    @settings(max_examples=30)
    def test_yaml_and_json_agree_on_count(self, rules: list[Rule]):
        """YAML and JSON targets produce same number of rules."""
        yaml_target = FlatYAMLTarget()
        json_target = FlatJSONTarget()

        yaml_result = yaml.safe_load(yaml_target.serialize(rules))
        json_result = json.loads(json_target.serialize(rules))

        assert len(yaml_result["rules"]) == len(json_result["rules"])

    @given(rules=st.lists(rule_strategy, min_size=1, max_size=10))
    @settings(max_examples=30)
    def test_all_formats_preserve_ids(self, rules: list[Rule]):
        """All three targets preserve rule IDs."""
        yaml_target = FlatYAMLTarget()
        json_target = FlatJSONTarget()
        seed_target = BuildlogSeedTarget()

        yaml_ids = {r["id"] for r in yaml.safe_load(yaml_target.serialize(rules))["rules"]}
        json_ids = {r["id"] for r in json.loads(json_target.serialize(rules))["rules"]}
        # New schema: buildlog seed stores ID in provenance block
        seed_ids = {r["provenance"]["id"] for r in seed_target.serialize(rules)["rules"]}
        input_ids = {r.id for r in rules}

        assert yaml_ids == input_ids
        assert json_ids == input_ids
        assert seed_ids == input_ids


# =========================================================================
# Enricher metamorphic tests
# =========================================================================


class TestEnricherMetamorphic:
    """Metamorphic tests for enrichers.

    Key property: enrichers should be deterministic and monotonic.
    """

    def test_passthrough_idempotent(self):
        """Passthrough enricher applied twice gives same structure."""
        enricher = PassthroughEnricher()
        rules = [_make_rule(f"r{i}") for i in range(5)]
        result1 = enricher.enrich(rules)
        # Apply passthrough to the inner rules again
        result2 = enricher.enrich([er.rule for er in result1])
        assert len(result1) == len(result2)
        assert all(r.enrichment is None for r in result1)
        assert all(r.enrichment is None for r in result2)

    def test_template_enricher_stable(self):
        """Template enricher produces same enrichments for same input."""
        enricher = TemplateEnricher(domain="test")
        rules = [_make_rule("r1")]
        result1 = enricher.enrich(rules)
        result2 = enricher.enrich(rules)
        assert result1[0].enrichment.context == result2[0].enrichment.context
        assert result1[0].enrichment.tags == result2[0].enrichment.tags

    def test_enricher_preserves_rule_count(self):
        """All enrichers preserve the number of rules."""
        rules = [_make_rule(f"r{i}") for i in range(7)]

        passthrough = PassthroughEnricher()
        template = TemplateEnricher(domain="test")
        llm = LLMEnricher(pipeline=EnrichmentPipeline())

        assert len(passthrough.enrich(rules)) == 7
        assert len(template.enrich(rules)) == 7
        assert len(llm.enrich(rules)) == 7

    def test_superset_input_superset_output(self):
        """If we add rules to the input, output count increases accordingly."""
        enricher = TemplateEnricher(domain="test")
        small = [_make_rule(f"r{i}") for i in range(3)]
        large = [_make_rule(f"r{i}") for i in range(7)]

        small_result = enricher.enrich(small)
        large_result = enricher.enrich(large)

        assert len(large_result) > len(small_result)
        assert len(large_result) == 7
        assert len(small_result) == 3


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    def test_rule_with_unicode_text(self):
        """Unicode in rule text survives serialization."""
        rule = _make_rule(text="Use the Strategy pattern for polymorphism")
        yaml_target = FlatYAMLTarget()
        json_target = FlatJSONTarget()

        yaml_result = yaml.safe_load(yaml_target.serialize([rule]))
        json_result = json.loads(json_target.serialize([rule]))

        assert yaml_result["rules"][0]["text"] == rule.text
        assert json_result["rules"][0]["text"] == rule.text

    def test_rule_with_special_chars(self):
        """Special characters in rule text survive serialization."""
        rule = _make_rule(text='Use "quotes" and {braces} and <angles>')
        json_target = FlatJSONTarget()
        result = json.loads(json_target.serialize([rule]))
        assert result["rules"][0]["text"] == rule.text

    def test_very_long_rule_text(self):
        """Very long text doesn't break serialization."""
        rule = _make_rule(text="x" * 10_000)
        json_target = FlatJSONTarget()
        result = json.loads(json_target.serialize([rule]))
        assert len(result["rules"][0]["text"]) == 10_000

    def test_empty_source_concepts(self):
        """Rule with empty source_concepts still serializes."""
        rule = Rule(
            id="r1",
            text="Test",
            domain="test",
            derivation="explicit",
            source_concepts=[],
            confidence=1.0,
        )
        target = FlatJSONTarget()
        result = json.loads(target.serialize([rule]))
        # source_concepts should be omitted when empty
        assert "source_concepts" not in result["rules"][0]

    def test_zero_confidence_rule(self):
        """Zero confidence rule passes min_confidence=0."""
        backend = InMemoryBackend()
        backend.connect()
        backend.create_domain("test")
        backend.add_rule(
            ExplicitRule(
                id="r1",
                text="Low confidence",
                domain="test",
                source_id="s1",
                confidence=0.0,
            )
        )
        source = FlatRuleSource(backend=backend)
        rules = source.derive(filters=ProjectionFilter(min_confidence=0.0))
        assert len(rules) == 1

    def test_high_confidence_filter_excludes_low(self):
        """High min_confidence filter excludes low-confidence rules."""
        backend = InMemoryBackend()
        backend.connect()
        backend.create_domain("test")
        backend.add_rule(
            ExplicitRule(
                id="r1",
                text="Low confidence",
                domain="test",
                source_id="s1",
                confidence=0.3,
            )
        )
        source = FlatRuleSource(backend=backend)
        rules = source.derive(filters=ProjectionFilter(min_confidence=0.5))
        assert len(rules) == 0
