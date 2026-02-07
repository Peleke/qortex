"""Tests for causal type system."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from qortex.causal.types import (
    RELATION_CAUSAL_DIRECTION,
    CausalAnnotation,
    CausalCapability,
    CausalDirection,
    CausalEdge,
    CausalNode,
    CausalQuery,
    CausalResult,
    CreditAssignment,
    IndependenceAssertion,
    QueryType,
)


class TestEnums:
    def test_causal_direction_values(self):
        assert CausalDirection.FORWARD == "forward"
        assert CausalDirection.REVERSE == "reverse"
        assert CausalDirection.BIDIRECTIONAL == "bidirectional"
        assert CausalDirection.NONE == "none"

    def test_causal_capability_values(self):
        assert CausalCapability.D_SEPARATION == "d_separation"
        assert CausalCapability.COUNTERFACTUAL == "counterfactual"

    def test_query_type_values(self):
        assert QueryType.OBSERVATIONAL == "observational"
        assert QueryType.INTERVENTIONAL == "interventional"
        assert QueryType.COUNTERFACTUAL == "counterfactual"

    def test_strenum_string_behavior(self):
        """StrEnum values can be compared with plain strings."""
        assert CausalDirection.FORWARD == "forward"
        assert str(CausalDirection.FORWARD) == "forward"


class TestDataclasses:
    def test_causal_node_defaults(self):
        node = CausalNode(id="n1", name="Node 1", domain="test")
        assert node.distribution_family is None
        assert node.parameter_priors is None
        assert node.properties == {}

    def test_causal_edge_defaults(self):
        edge = CausalEdge(
            source_id="a",
            target_id="b",
            relation_type="requires",
            direction=CausalDirection.FORWARD,
        )
        assert edge.strength == 1.0
        assert edge.functional_form is None
        assert edge.learned_params is None

    def test_causal_annotation_frozen(self):
        ann = CausalAnnotation(
            direction=CausalDirection.FORWARD,
            strength=0.9,
        )
        # Frozen — should be hashable
        assert hash(ann) is not None

    def test_causal_annotation_roundtrip(self):
        ann = CausalAnnotation(
            direction=CausalDirection.FORWARD,
            strength=0.9,
            functional_form="linear",
        )
        d = ann.to_dict()
        restored = CausalAnnotation.from_dict(d)
        assert restored == ann

    def test_causal_query_defaults(self):
        q = CausalQuery(
            query_type=QueryType.OBSERVATIONAL,
            target_nodes=frozenset({"X"}),
        )
        assert q.conditioning_nodes == frozenset()
        assert q.intervention_nodes == frozenset()
        assert q.intervention_values == {}

    def test_causal_result_defaults(self):
        q = CausalQuery(query_type=QueryType.OBSERVATIONAL, target_nodes=frozenset({"X"}))
        r = CausalResult(query=q)
        assert r.independences == []
        assert r.causal_effect is None
        assert r.effect_bounds is None

    def test_independence_assertion(self):
        a = IndependenceAssertion(
            x=frozenset({"A"}),
            y=frozenset({"C"}),
            z=frozenset({"B"}),
            is_independent=True,
            method="d_separation",
        )
        assert a.p_value is None
        assert a.confidence == 1.0

    def test_credit_assignment(self):
        ca = CreditAssignment(
            concept_id="c1",
            credit=0.5,
            path=["c1"],
            method="direct",
        )
        assert ca.credit == 0.5


class TestRelationMapping:
    def test_all_relation_types_mapped(self):
        expected = {
            "requires",
            "implements",
            "refines",
            "part_of",
            "uses",
            "supports",
            "challenges",
            "contradicts",
            "similar_to",
            "alternative_to",
        }
        assert set(RELATION_CAUSAL_DIRECTION.keys()) == expected

    def test_forward_relations(self):
        for rel in ("requires", "uses", "supports", "challenges"):
            direction, _ = RELATION_CAUSAL_DIRECTION[rel]
            assert direction == CausalDirection.FORWARD

    def test_reverse_relations(self):
        for rel in ("implements", "refines", "part_of"):
            direction, _ = RELATION_CAUSAL_DIRECTION[rel]
            assert direction == CausalDirection.REVERSE

    def test_excluded_relations(self):
        direction, _ = RELATION_CAUSAL_DIRECTION["contradicts"]
        assert direction == CausalDirection.BIDIRECTIONAL

        for rel in ("similar_to", "alternative_to"):
            direction, _ = RELATION_CAUSAL_DIRECTION[rel]
            assert direction == CausalDirection.NONE

    def test_confidence_range(self):
        for _, (_, conf) in RELATION_CAUSAL_DIRECTION.items():
            assert 0.0 < conf <= 1.0


class TestAnnotationErrorPaths:
    """CausalAnnotation.from_dict() error handling."""

    def test_missing_direction_key(self):
        with pytest.raises(ValueError, match="missing 'direction'"):
            CausalAnnotation.from_dict({"strength": 0.9})

    def test_missing_strength_key(self):
        with pytest.raises(ValueError, match="missing 'strength'"):
            CausalAnnotation.from_dict({"direction": "forward"})

    def test_invalid_direction_value(self):
        with pytest.raises(ValueError):
            CausalAnnotation.from_dict({"direction": "bogus", "strength": 0.5})

    def test_non_numeric_strength(self):
        with pytest.raises(TypeError, match="strength must be numeric"):
            CausalAnnotation.from_dict({"direction": "forward", "strength": "high"})

    def test_nan_strength(self):
        with pytest.raises(ValueError, match="strength must be finite"):
            CausalAnnotation.from_dict({"direction": "forward", "strength": float("nan")})

    def test_inf_strength(self):
        with pytest.raises(ValueError, match="strength must be finite"):
            CausalAnnotation.from_dict({"direction": "forward", "strength": float("inf")})


class TestAnnotationPropertyBased:
    """Property-based roundtrip test for CausalAnnotation."""

    @given(
        direction=st.sampled_from(list(CausalDirection)),
        strength=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        functional_form=st.one_of(st.none(), st.text(min_size=0, max_size=50)),
    )
    def test_roundtrip(self, direction, strength, functional_form):
        ann = CausalAnnotation(
            direction=direction,
            strength=strength,
            functional_form=functional_form,
        )
        restored = CausalAnnotation.from_dict(ann.to_dict())
        assert restored == ann


class TestEdgeStrengthValidation:
    """CausalEdge strength validation."""

    def test_nan_strength_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            CausalEdge("a", "b", "r", CausalDirection.FORWARD, strength=float("nan"))

    def test_inf_strength_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            CausalEdge("a", "b", "r", CausalDirection.FORWARD, strength=float("inf"))

    def test_negative_strength_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            CausalEdge("a", "b", "r", CausalDirection.FORWARD, strength=-0.1)

    def test_zero_strength_accepted(self):
        edge = CausalEdge("a", "b", "r", CausalDirection.FORWARD, strength=0.0)
        assert edge.strength == 0.0
