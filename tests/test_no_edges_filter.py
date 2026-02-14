"""Tests for include_edge_derived filter on ProjectionFilter and FlatRuleSource."""

import pytest

from qortex.core.models import ConceptEdge, ConceptNode, RelationType, Rule
from qortex.projectors.models import ProjectionFilter
from qortex.projectors.sources.flat import FlatRuleSource


# ---------------------------------------------------------------------------
# Minimal in-memory backend stub
# ---------------------------------------------------------------------------


class _StubBackend:
    """Minimal backend for testing FlatRuleSource edge derivation."""

    def __init__(self, nodes=None, edges=None, rules=None, domains=None):
        self._nodes = {n.id: n for n in (nodes or [])}
        self._edges = edges or {}  # node_id -> list[ConceptEdge]
        self._rules = rules or {}  # domain -> list
        self._domains = domains or []

    def find_nodes(self, domain=None, limit=10_000):
        return [n for n in self._nodes.values() if domain is None or n.domain == domain]

    def get_node(self, node_id):
        return self._nodes.get(node_id)

    def get_edges(self, node_id, direction="out"):
        return self._edges.get(node_id, [])

    def get_rules(self, domain):
        return self._rules.get(domain, [])

    def list_domains(self):
        return self._domains

    # GraphBackend protocol requires connect/close
    def connect(self):
        pass

    def close(self):
        pass


class _StubDomain:
    def __init__(self, name):
        self.name = name


class _StubExplicitRule:
    def __init__(self, id, text, domain, confidence=0.9, category="principle"):
        self.id = id
        self.text = text
        self.domain = domain
        self.confidence = confidence
        self.category = category
        self.concept_ids = [f"{domain}:c1"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_backend_with_edges():
    """Backend with 2 explicit rules and 2 edge-derivable pairs."""
    domain = "test_domain"
    n1 = ConceptNode(
        id=f"{domain}:Publisher", name="Publisher",
        description="A publisher component", domain=domain, source_id="test",
    )
    n2 = ConceptNode(
        id=f"{domain}:Subscriber", name="Subscriber",
        description="A subscriber component", domain=domain, source_id="test",
    )

    edge = ConceptEdge(
        source_id=n1.id,
        target_id=n2.id,
        relation_type=RelationType.USES,
        confidence=0.8,
    )

    explicit_rules = [
        _StubExplicitRule(
            id=f"{domain}:rule:0",
            text="The publisher doesn't know how subscribers are implemented.",
            domain=domain,
        ),
        _StubExplicitRule(
            id=f"{domain}:rule:1",
            text="Subscribers can join and leave at any time.",
            domain=domain,
        ),
    ]

    return _StubBackend(
        nodes=[n1, n2],
        edges={n1.id: [edge]},
        rules={domain: explicit_rules},
        domains=[_StubDomain(domain)],
    )


# ---------------------------------------------------------------------------
# Tests: ProjectionFilter
# ---------------------------------------------------------------------------


class TestProjectionFilterIncludeEdgeDerived:
    def test_default_is_true(self):
        f = ProjectionFilter()
        assert f.include_edge_derived is True

    def test_can_set_false(self):
        f = ProjectionFilter(include_edge_derived=False)
        assert f.include_edge_derived is False

    def test_backward_compatible_with_existing_fields(self):
        f = ProjectionFilter(
            domains=["a"],
            categories=["arch"],
            min_confidence=0.5,
            derivation="explicit",
            relation_types=[RelationType.REQUIRES],
            include_edge_derived=False,
        )
        assert f.domains == ["a"]
        assert f.include_edge_derived is False


# ---------------------------------------------------------------------------
# Tests: FlatRuleSource with include_edge_derived filter
# ---------------------------------------------------------------------------


class TestFlatRuleSourceNoEdges:
    def test_default_includes_edge_derived(self):
        """Default filter includes both explicit and edge-derived rules."""
        backend = _make_backend_with_edges()
        source = FlatRuleSource(backend=backend)
        rules = source.derive()
        explicit = [r for r in rules if r.derivation == "explicit"]
        derived = [r for r in rules if r.derivation == "derived"]
        assert len(explicit) == 2
        assert len(derived) >= 1

    def test_no_edges_excludes_derived(self):
        """include_edge_derived=False excludes all edge-derived rules."""
        backend = _make_backend_with_edges()
        source = FlatRuleSource(backend=backend)
        filt = ProjectionFilter(include_edge_derived=False)
        rules = source.derive(filters=filt)
        derivations = {r.derivation for r in rules}
        assert "derived" not in derivations
        assert all(r.derivation == "explicit" for r in rules)

    def test_no_edges_keeps_explicit(self):
        """Explicit rules survive the edge filter."""
        backend = _make_backend_with_edges()
        source = FlatRuleSource(backend=backend)
        filt = ProjectionFilter(include_edge_derived=False)
        rules = source.derive(filters=filt)
        assert len(rules) == 2
        texts = {r.text for r in rules}
        assert "The publisher doesn't know how subscribers are implemented." in texts
        assert "Subscribers can join and leave at any time." in texts

    def test_no_edges_returns_empty_for_derived_only(self):
        """If derivation='derived' + no edges, result is empty."""
        backend = _make_backend_with_edges()
        source = FlatRuleSource(backend=backend)
        filt = ProjectionFilter(derivation="derived", include_edge_derived=False)
        rules = source.derive(filters=filt)
        assert rules == []

    def test_edges_enabled_includes_template_metadata(self):
        """Edge-derived rules have template_id in metadata."""
        backend = _make_backend_with_edges()
        source = FlatRuleSource(backend=backend)
        filt = ProjectionFilter(include_edge_derived=True)
        rules = source.derive(filters=filt)
        derived = [r for r in rules if r.derivation == "derived"]
        assert len(derived) >= 1
        for r in derived:
            assert r.metadata.get("template_id") is not None
            assert r.metadata.get("relation_type") is not None


# ---------------------------------------------------------------------------
# Tests: Projection pipeline with no-edges filter
# ---------------------------------------------------------------------------


class TestProjectionPipelineNoEdges:
    def test_filter_passes_through_projection(self):
        """ProjectionFilter.include_edge_derived flows through Projection.project()."""
        from qortex.projectors.projection import Projection
        from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget

        backend = _make_backend_with_edges()
        source = FlatRuleSource(backend=backend)
        target = BuildlogSeedTarget(persona_name="test")

        projection = Projection(source=source, target=target)

        # With edges
        result_with = projection.project(
            filters=ProjectionFilter(include_edge_derived=True),
        )
        # Without edges
        result_without = projection.project(
            filters=ProjectionFilter(include_edge_derived=False),
        )

        assert result_with["metadata"]["rule_count"] > result_without["metadata"]["rule_count"]
        assert result_without["metadata"]["rule_count"] == 2

        # Verify no edge-derived rules in the output
        for rule_entry in result_without["rules"]:
            prov = rule_entry.get("provenance", {})
            assert prov.get("template_id") is None

    def test_seed_output_only_explicit_rules(self):
        """Buildlog seed with --no-edges contains only explicit content."""
        from qortex.projectors.targets.buildlog_seed import BuildlogSeedTarget

        backend = _make_backend_with_edges()
        source = FlatRuleSource(backend=backend)
        target = BuildlogSeedTarget(persona_name="test_clean")

        from qortex.projectors.projection import Projection

        projection = Projection(source=source, target=target)
        result = projection.project(
            filters=ProjectionFilter(include_edge_derived=False),
        )

        assert result["persona"] == "test_clean"
        assert len(result["rules"]) == 2
        for rule_entry in result["rules"]:
            # Should be real content, not "X uses Y, creating coupling"
            assert "creating coupling" not in rule_entry["rule"]
            assert "is a concrete way to achieve" not in rule_entry["rule"]
