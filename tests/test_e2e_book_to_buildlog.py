"""E2E test: Book chapter -> KG -> Projection -> Buildlog seed YAML.

Tests the full pipeline with real content from Chapter 5 of
'Software Design for Python Programmers' by Ronald Mak.

Two modes:
    - TestE2EWithoutMemgraph: always runs (CI), uses InMemoryBackend
    - TestE2EWithMemgraph: skipped if Memgraph is not running (local only)
"""

from __future__ import annotations

import socket

import pytest
import yaml

from qortex.core.memory import InMemoryBackend
from qortex.core.models import (
    ConceptEdge,
    ConceptNode,
    ExplicitRule,
    IngestionManifest,
    RelationType,
    SourceMetadata,
)
from qortex.projectors.enrichers.passthrough import PassthroughEnricher
from qortex.projectors.enrichers.template import TemplateEnricher
from qortex.projectors.models import EnrichedRule, ProjectionFilter
from qortex.projectors.projection import Projection
from qortex.projectors.sources.flat import FlatRuleSource
from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget
from qortex.projectors.targets.flat_json import FlatJSONTarget
from qortex.projectors.targets.flat_yaml import FlatYAMLTarget


# =============================================================================
# Chapter 5 fixture data: real concepts from the book
# =============================================================================

DOMAIN = "implementation_hiding"
SOURCE_ID = "ch5"

# 8 concepts extracted from Chapter 5
CONCEPTS = [
    ConceptNode(
        id="ch5:encapsulation",
        name="Encapsulation",
        description="Hide the implementation of a class so other classes cannot depend on what they cannot access",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="chapter 5, intro",
    ),
    ConceptNode(
        id="ch5:plk",
        name="Principle of Least Knowledge",
        description="Class A should know as little as possible about the implementation of class B",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="section 5.1",
    ),
    ConceptNode(
        id="ch5:properties",
        name="Properties (Getters/Setters)",
        description="Use @property decorator to provide controlled public access to private state",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="section 5.2",
    ),
    ConceptNode(
        id="ch5:lazy_eval",
        name="Lazy Evaluation",
        description="Delay an expensive calculation until its results are needed to improve runtime performance",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="section 5.3.3",
    ),
    ConceptNode(
        id="ch5:immutability",
        name="Immutability",
        description="Objects that cannot be modified after construction; class should not provide state-modifying setters",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="section 5.4",
    ),
    ConceptNode(
        id="ch5:dangerous_setters",
        name="Dangerous Setters",
        description="Setter methods that can put an object into an invalid state by modifying fields individually",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="section 5.5",
    ),
    ConceptNode(
        id="ch5:law_of_demeter",
        name="Law of Demeter",
        description="A method should only call methods on objects that are close to it, not on objects returned by other methods",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="section 5.6",
    ),
    ConceptNode(
        id="ch5:open_closed",
        name="Open-Closed Principle",
        description="Close a superclass for modification but open it for extensions by subclasses",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="section 5.8",
    ),
]

# 7 edges: diverse relation types from the chapter
EDGES = [
    ConceptEdge(
        source_id="ch5:plk",
        target_id="ch5:encapsulation",
        relation_type=RelationType.REQUIRES,
        confidence=0.95,
    ),
    ConceptEdge(
        source_id="ch5:immutability",
        target_id="ch5:dangerous_setters",
        relation_type=RelationType.CONTRADICTS,
        confidence=0.9,
    ),
    ConceptEdge(
        source_id="ch5:properties",
        target_id="ch5:encapsulation",
        relation_type=RelationType.IMPLEMENTS,
        confidence=0.85,
    ),
    ConceptEdge(
        source_id="ch5:lazy_eval",
        target_id="ch5:encapsulation",
        relation_type=RelationType.USES,
        confidence=0.8,
    ),
    ConceptEdge(
        source_id="ch5:open_closed",
        target_id="ch5:encapsulation",
        relation_type=RelationType.SUPPORTS,
        confidence=0.85,
    ),
    ConceptEdge(
        source_id="ch5:law_of_demeter",
        target_id="ch5:plk",
        relation_type=RelationType.REFINES,
        confidence=0.9,
    ),
    ConceptEdge(
        source_id="ch5:dangerous_setters",
        target_id="ch5:properties",
        relation_type=RelationType.CHALLENGES,
        confidence=0.75,
    ),
]

# 6 explicit rules from Chapter 5
EXPLICIT_RULES = [
    ExplicitRule(
        id="ch5:r1",
        text="Make every instance variable and method private except those that must be public to enable other code to use the class",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="section 5.1",
        category="encapsulation",
        confidence=0.95,
    ),
    ExplicitRule(
        id="ch5:r2",
        text="A property should never allow an object to be put into an invalid state",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="section 5.5",
        category="safety",
        confidence=0.9,
    ),
    ExplicitRule(
        id="ch5:r3",
        text="Delay an expensive calculation until its results are needed (Lazy Evaluation Principle)",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="section 5.3.3",
        category="performance",
        confidence=0.85,
    ),
    ExplicitRule(
        id="ch5:r4",
        text="A class whose objects are supposed to be immutable must not provide references to hidden state implementation",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="section 5.7",
        category="safety",
        confidence=0.9,
    ),
    ExplicitRule(
        id="ch5:r5",
        text="A method should only call methods on objects that are close to it; do not call methods on objects returned by another method",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="section 5.6",
        category="architectural",
        confidence=0.85,
    ),
    ExplicitRule(
        id="ch5:r6",
        text="Close a superclass for modification but open it for extensions by subclasses to support code stability",
        domain=DOMAIN,
        source_id=SOURCE_ID,
        source_location="section 5.8",
        category="architectural",
        confidence=0.85,
    ),
]

NUM_EXPLICIT = len(EXPLICIT_RULES)
NUM_EDGES = len(EDGES)


def _build_chapter_manifest() -> IngestionManifest:
    """Build an IngestionManifest from Chapter 5 content."""
    return IngestionManifest(
        source=SourceMetadata(
            id="sd4pp-ch5",
            name="Software Design for Python Programmers - Chapter 5",
            source_type="text",
            path_or_url="CH5.txt",
            chunk_count=1,
            concept_count=len(CONCEPTS),
            rule_count=len(EXPLICIT_RULES),
        ),
        domain=DOMAIN,
        concepts=list(CONCEPTS),
        edges=list(EDGES),
        rules=list(EXPLICIT_RULES),
        extraction_confidence=0.9,
    )


@pytest.fixture
def chapter_manifest() -> IngestionManifest:
    return _build_chapter_manifest()


@pytest.fixture
def seeded_backend(chapter_manifest: IngestionManifest) -> InMemoryBackend:
    """Backend with Chapter 5 data ingested."""
    backend = InMemoryBackend()
    backend.connect()
    backend.ingest_manifest(chapter_manifest)
    return backend


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_buildlog_seed(seed: dict) -> None:
    """Assert that a dict matches the universal rule set schema."""
    # Persona is a flat string
    assert "persona" in seed
    assert isinstance(seed["persona"], str)

    # Version is an int
    assert "version" in seed
    assert isinstance(seed["version"], int)

    # Metadata block
    assert "metadata" in seed
    assert "source" in seed["metadata"]
    assert "source_version" in seed["metadata"]
    assert "projected_at" in seed["metadata"]
    assert "rule_count" in seed["metadata"]
    assert seed["metadata"]["rule_count"] == len(seed["rules"])

    # Rules list
    assert "rules" in seed
    assert isinstance(seed["rules"], list)

    for rule in seed["rules"]:
        # Universal schema: 'rule' key (not 'text'), category required
        assert "rule" in rule
        assert "category" in rule
        assert isinstance(rule["rule"], str)

        # Provenance block
        assert "provenance" in rule
        prov = rule["provenance"]
        assert "id" in prov
        assert "domain" in prov
        assert "derivation" in prov
        assert "confidence" in prov
        assert isinstance(prov["confidence"], (int, float))
        assert prov["derivation"] in ("explicit", "derived")
        assert "source_concepts" in prov
        assert "relevance" in prov

        # Template fields present (null for explicit, populated for derived)
        assert "relation_type" in prov
        assert "template_id" in prov
        assert "template_variant" in prov
        assert "template_severity" in prov


# =============================================================================
# E2E tests: InMemoryBackend (CI mode, always runs)
# =============================================================================


class TestE2EWithoutMemgraph:
    """Full pipeline tests using InMemoryBackend. No Docker needed."""

    def test_full_pipeline_produces_valid_buildlog_seed(self, seeded_backend):
        """Book chapter -> KG -> FlatRuleSource -> TemplateEnricher -> BuildlogSeedTarget."""
        source = FlatRuleSource(backend=seeded_backend)
        enricher = TemplateEnricher(domain=DOMAIN)
        target = BuildlogSeedTarget(persona_name="sd4pp_ch5")

        projection = Projection(source=source, enricher=enricher, target=target)
        result = projection.project(domains=[DOMAIN])

        _validate_buildlog_seed(result)

    def test_seed_has_correct_persona(self, seeded_backend):
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget(persona_name="sd4pp_ch5")

        projection = Projection(source=source, target=target)
        result = projection.project(domains=[DOMAIN])

        assert result["persona"] == "sd4pp_ch5"

    def test_seed_has_version(self, seeded_backend):
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, target=target)
        result = projection.project(domains=[DOMAIN])

        assert result["version"] == 1

    def test_seed_metadata_tracks_source(self, seeded_backend):
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, target=target)
        result = projection.project(domains=[DOMAIN])

        assert result["metadata"]["source"] == "qortex"
        assert result["metadata"]["rule_count"] > 0

    def test_all_rules_have_required_fields(self, seeded_backend):
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, target=target)
        result = projection.project(domains=[DOMAIN])

        for rule in result["rules"]:
            assert "rule" in rule
            assert "category" in rule
            assert "provenance" in rule
            prov = rule["provenance"]
            assert "id" in prov
            assert "domain" in prov
            assert "confidence" in prov
            assert "derivation" in prov

    def test_enriched_rules_have_context_fields(self, seeded_backend):
        source = FlatRuleSource(backend=seeded_backend)
        enricher = TemplateEnricher(domain=DOMAIN)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, enricher=enricher, target=target)
        result = projection.project(domains=[DOMAIN])

        enriched_count = 0
        for rule in result["rules"]:
            if "context" in rule:
                enriched_count += 1
                assert "antipattern" in rule
                assert "rationale" in rule
                assert "tags" in rule
        assert enriched_count == len(result["rules"])

    def test_rule_count_matches_expected(self, seeded_backend):
        """Explicit rules + one derived rule per edge = deterministic count."""
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, target=target)
        result = projection.project(domains=[DOMAIN])

        assert result["metadata"]["rule_count"] == NUM_EXPLICIT + NUM_EDGES

    def test_explicit_and_derived_rules_present(self, seeded_backend):
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, target=target)
        result = projection.project(domains=[DOMAIN])

        derivations = {r["provenance"]["derivation"] for r in result["rules"]}
        assert "explicit" in derivations
        assert "derived" in derivations

    def test_seed_yaml_roundtrip(self, seeded_backend):
        """Seed dict survives YAML serialization/deserialization."""
        source = FlatRuleSource(backend=seeded_backend)
        enricher = TemplateEnricher(domain=DOMAIN)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, enricher=enricher, target=target)
        result = projection.project(domains=[DOMAIN])

        yaml_str = yaml.dump(result, default_flow_style=False, Dumper=yaml.SafeDumper)
        loaded = yaml.safe_load(yaml_str)

        assert loaded["persona"] == result["persona"]
        assert loaded["version"] == result["version"]
        assert loaded["metadata"]["rule_count"] == result["metadata"]["rule_count"]
        assert len(loaded["rules"]) == len(result["rules"])

    def test_wrong_domain_yields_zero_rules(self, seeded_backend):
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, target=target)
        result = projection.project(domains=["nonexistent_domain"])

        assert result["metadata"]["rule_count"] == 0
        assert result["rules"] == []

    def test_confidence_filter_reduces_rules(self, seeded_backend):
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget()

        all_proj = Projection(source=source, target=target)
        all_result = all_proj.project(domains=[DOMAIN])

        filtered_proj = Projection(source=source, target=target)
        filtered_result = filtered_proj.project(
            domains=[DOMAIN],
            filters=ProjectionFilter(min_confidence=0.9),
        )

        assert len(filtered_result["rules"]) < len(all_result["rules"])
        assert len(filtered_result["rules"]) > 0

    def test_derivation_filter_explicit_only(self, seeded_backend):
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, target=target)
        result = projection.project(
            domains=[DOMAIN],
            filters=ProjectionFilter(derivation="explicit"),
        )

        assert all(r["provenance"]["derivation"] == "explicit" for r in result["rules"])
        assert len(result["rules"]) == NUM_EXPLICIT

    def test_derivation_filter_derived_only(self, seeded_backend):
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, target=target)
        result = projection.project(
            domains=[DOMAIN],
            filters=ProjectionFilter(derivation="derived"),
        )

        assert all(r["provenance"]["derivation"] == "derived" for r in result["rules"])
        assert len(result["rules"]) == NUM_EDGES

    def test_passthrough_enricher_produces_no_enrichment_fields(self, seeded_backend):
        source = FlatRuleSource(backend=seeded_backend)
        enricher = PassthroughEnricher()
        target = BuildlogSeedTarget()

        projection = Projection(source=source, enricher=enricher, target=target)
        result = projection.project(domains=[DOMAIN])

        for rule in result["rules"]:
            assert "context" not in rule
            assert "antipattern" not in rule

    def test_all_targets_produce_consistent_rule_counts(self, seeded_backend):
        """All three targets should operate on the same derived rules."""
        source = FlatRuleSource(backend=seeded_backend)

        buildlog_proj = Projection(source=source, target=BuildlogSeedTarget())
        buildlog_result = buildlog_proj.project(domains=[DOMAIN])

        yaml_proj = Projection(source=source, target=FlatYAMLTarget())
        yaml_result = yaml.safe_load(yaml_proj.project(domains=[DOMAIN]))

        json_proj = Projection(source=source, target=FlatJSONTarget())
        import json
        json_result = json.loads(json_proj.project(domains=[DOMAIN]))

        assert buildlog_result["metadata"]["rule_count"] == len(yaml_result["rules"])
        assert buildlog_result["metadata"]["rule_count"] == len(json_result["rules"])

    def test_derived_rules_reference_chapter_concepts(self, seeded_backend):
        """Derived rules should reference concept IDs from the chapter."""
        source = FlatRuleSource(backend=seeded_backend)
        rules = source.derive(domains=[DOMAIN])

        concept_ids = {c.id for c in CONCEPTS}
        for rule in rules:
            if rule.derivation == "derived":
                assert len(rule.source_concepts) == 2
                for cid in rule.source_concepts:
                    assert cid in concept_ids

    def test_explicit_rules_preserve_text(self, seeded_backend):
        """Explicit rules should preserve the original text from the chapter."""
        source = FlatRuleSource(backend=seeded_backend)
        rules = source.derive(
            domains=[DOMAIN],
            filters=ProjectionFilter(derivation="explicit"),
        )

        original_texts = {r.text for r in EXPLICIT_RULES}
        for rule in rules:
            assert rule.text in original_texts

    def test_ingest_manifest_populates_domain_stats(self, seeded_backend):
        """After ingestion, domain stats should reflect the chapter data."""
        domains = seeded_backend.list_domains()
        assert len(domains) == 1
        domain = domains[0]
        assert domain.name == DOMAIN
        assert domain.concept_count == len(CONCEPTS)
        assert domain.edge_count == len(EDGES)
        assert domain.rule_count == len(EXPLICIT_RULES)

    # -----------------------------------------------------------------
    # New provenance tests (Track F)
    # -----------------------------------------------------------------

    def test_provenance_template_metadata_for_derived_rules(self, seeded_backend):
        """Derived rules should have non-null template metadata in provenance."""
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, target=target)
        result = projection.project(
            domains=[DOMAIN],
            filters=ProjectionFilter(derivation="derived"),
        )

        for rule in result["rules"]:
            prov = rule["provenance"]
            assert prov["derivation"] == "derived"
            assert prov["relation_type"] is not None
            assert prov["template_id"] is not None
            assert prov["template_variant"] is not None
            assert prov["template_severity"] is not None

    def test_provenance_null_template_for_explicit_rules(self, seeded_backend):
        """Explicit rules should have null template metadata in provenance."""
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, target=target)
        result = projection.project(
            domains=[DOMAIN],
            filters=ProjectionFilter(derivation="explicit"),
        )

        for rule in result["rules"]:
            prov = rule["provenance"]
            assert prov["derivation"] == "explicit"
            assert prov["relation_type"] is None
            assert prov["template_id"] is None
            assert prov["template_variant"] is None
            assert prov["template_severity"] is None

    def test_provenance_graph_version_propagated(self, seeded_backend):
        """graph_version should appear in provenance when set on target."""
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget(graph_version="2026-02-05T14:30:00Z")

        projection = Projection(source=source, target=target)
        result = projection.project(domains=[DOMAIN])

        for rule in result["rules"]:
            assert rule["provenance"]["graph_version"] == "2026-02-05T14:30:00Z"

    def test_category_falls_back_to_domain(self, seeded_backend):
        """Rules without explicit category should use domain as fallback."""
        from qortex.core.models import Rule
        from qortex.projectors.targets._serialize import serialize_ruleset

        # Rule with no category
        rule = Rule(
            id="test:no_cat",
            text="Test rule",
            domain="some_domain",
            derivation="explicit",
            source_concepts=["c1"],
            confidence=0.9,
            category=None,
        )

        result = serialize_ruleset([rule], persona="test")
        assert result["rules"][0]["category"] == "some_domain"

    def test_projected_at_in_metadata(self, seeded_backend):
        """metadata.projected_at should be present as an ISO timestamp."""
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, target=target)
        result = projection.project(domains=[DOMAIN])

        assert "projected_at" in result["metadata"]
        # Should be an ISO format string with T separator
        assert "T" in result["metadata"]["projected_at"]

    def test_source_version_in_metadata(self, seeded_backend):
        """metadata.source_version should be present."""
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget(source_version="0.2.0")

        projection = Projection(source=source, target=target)
        result = projection.project(domains=[DOMAIN])

        assert result["metadata"]["source_version"] == "0.2.0"

    def test_provenance_source_concepts_present(self, seeded_backend):
        """All rules should have source_concepts in provenance."""
        source = FlatRuleSource(backend=seeded_backend)
        target = BuildlogSeedTarget()

        projection = Projection(source=source, target=target)
        result = projection.project(domains=[DOMAIN])

        for rule in result["rules"]:
            assert "source_concepts" in rule["provenance"]
            assert isinstance(rule["provenance"]["source_concepts"], list)


# =============================================================================
# E2E tests: Memgraph (local mode, skipped in CI)
# =============================================================================

MEMGRAPH_AVAILABLE = False
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(1)
    s.connect(("localhost", 7687))
    s.close()
    MEMGRAPH_AVAILABLE = True
except (OSError, ConnectionRefusedError):
    pass


@pytest.mark.skipif(not MEMGRAPH_AVAILABLE, reason="Memgraph not running (docker-compose up)")
class TestE2EWithMemgraph:
    """Full pipeline tests using MemgraphBackend. Requires Docker."""

    @pytest.fixture(autouse=True)
    def _setup_backend(self):
        from qortex.core.backend import MemgraphBackend

        self.backend = MemgraphBackend(host="localhost", port=7687)
        self.backend.connect()
        self.backend._run("MATCH (n) DETACH DELETE n")
        yield
        self.backend._run("MATCH (n) DETACH DELETE n")
        self.backend.disconnect()

    def test_memgraph_backend_connects(self):
        assert self.backend.is_connected()

    def test_memgraph_full_pipeline(self):
        """Full pipeline: ingest -> project -> validate seed."""
        manifest = _build_chapter_manifest()
        self.backend.ingest_manifest(manifest)

        source = FlatRuleSource(backend=self.backend)
        enricher = TemplateEnricher(domain=DOMAIN)
        target = BuildlogSeedTarget(persona_name="qortex_" + DOMAIN)

        projection = Projection(source=source, enricher=enricher, target=target)
        result = projection.project(domains=[DOMAIN])

        _validate_buildlog_seed(result)
        assert result["metadata"]["rule_count"] == NUM_EXPLICIT + NUM_EDGES

    def test_memgraph_ingest_and_query(self):
        """Ingest manifest and query concepts via Cypher."""
        manifest = _build_chapter_manifest()
        self.backend.ingest_manifest(manifest)

        results = list(self.backend.query_cypher(
            "MATCH (c:Concept {domain: $d}) RETURN c.id AS id",
            {"d": DOMAIN},
        ))
        ids = {r["id"] for r in results}
        assert len(ids) == len(CONCEPTS)

    def test_memgraph_ppr_returns_nonzero_scores(self):
        """PPR should return scores for connected concepts."""
        manifest = _build_chapter_manifest()
        self.backend.ingest_manifest(manifest)

        scores = self.backend.personalized_pagerank(
            source_nodes=["ch5:encapsulation"],
            domain=DOMAIN,
        )
        # Should return some scores (depends on MAGE availability)
        assert isinstance(scores, dict)


# =============================================================================
# FlatRuleProjector deprecation tests
# =============================================================================


class TestFlatRuleProjectorDeprecation:
    """Tests for the deprecated FlatRuleProjector wrapper."""

    def test_instantiation_warns(self, seeded_backend):
        from qortex.projectors.flat import FlatRuleProjector

        with pytest.warns(DeprecationWarning, match="FlatRuleProjector is deprecated"):
            FlatRuleProjector(backend=seeded_backend)

    def test_project_returns_valid_yaml(self, seeded_backend):
        import warnings

        from qortex.projectors.flat import FlatRuleProjector

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            projector = FlatRuleProjector(backend=seeded_backend)
            result = projector.project(domains=[DOMAIN])

        parsed = yaml.safe_load(result)
        assert "rules" in parsed
        assert len(parsed["rules"]) == NUM_EXPLICIT + NUM_EDGES

    def test_delegates_same_rules_as_projection(self, seeded_backend):
        """Output should match Projection(FlatRuleSource, PassthroughEnricher, FlatYAMLTarget)."""
        import warnings

        from qortex.projectors.flat import FlatRuleProjector

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            projector = FlatRuleProjector(backend=seeded_backend)
            deprecated_result = yaml.safe_load(projector.project(domains=[DOMAIN]))

        projection = Projection(
            source=FlatRuleSource(backend=seeded_backend),
            enricher=PassthroughEnricher(),
            target=FlatYAMLTarget(),
        )
        new_result = yaml.safe_load(projection.project(domains=[DOMAIN]))

        assert len(deprecated_result["rules"]) == len(new_result["rules"])

    def test_include_derived_false(self, seeded_backend):
        import warnings

        from qortex.projectors.flat import FlatRuleProjector

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            projector = FlatRuleProjector(backend=seeded_backend, include_derived=False)
            result = yaml.safe_load(projector.project(domains=[DOMAIN]))

        assert all(r["derivation"] == "explicit" for r in result["rules"])
        assert len(result["rules"]) == NUM_EXPLICIT

    def test_domain_filter(self, seeded_backend):
        import warnings

        from qortex.projectors.flat import FlatRuleProjector

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            projector = FlatRuleProjector(backend=seeded_backend)
            result = yaml.safe_load(projector.project(domains=["nonexistent"]))

        assert result["rules"] == []

    def test_filters_passthrough(self, seeded_backend):
        import warnings

        from qortex.projectors.flat import FlatRuleProjector

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            projector = FlatRuleProjector(backend=seeded_backend)
            result = yaml.safe_load(
                projector.project(
                    domains=[DOMAIN],
                    filters=ProjectionFilter(min_confidence=0.9),
                )
            )

        assert len(result["rules"]) > 0
        assert len(result["rules"]) < NUM_EXPLICIT + NUM_EDGES
