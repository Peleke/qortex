"""Causal type system — enums, dataclasses, relation mapping.

Every other causal module imports from here. Pyro-aware: optional fields
(distribution_family, parameter_priors, etc.) default to None so that
Phase 2 can populate them without rewriting dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# =============================================================================
# Enums
# =============================================================================


class CausalDirection(StrEnum):
    """Direction of causal influence along an edge."""

    FORWARD = "forward"
    REVERSE = "reverse"
    BIDIRECTIONAL = "bidirectional"
    NONE = "none"


class CausalCapability(StrEnum):
    """What a causal backend can do."""

    D_SEPARATION = "d_separation"
    CI_TESTING = "ci_testing"
    PROBABILISTIC_INFERENCE = "probabilistic_inference"
    INTERVENTION = "intervention"
    COUNTERFACTUAL = "counterfactual"


class QueryType(StrEnum):
    """Pearl's three rungs of the causal ladder."""

    OBSERVATIONAL = "observational"
    INTERVENTIONAL = "interventional"
    COUNTERFACTUAL = "counterfactual"


# =============================================================================
# Dataclasses
# =============================================================================


@dataclass
class CausalNode:
    """A node in the causal DAG."""

    id: str
    name: str
    domain: str
    properties: dict[str, Any] = field(default_factory=dict)

    # Pyro Phase 2: distribution family for sample sites
    distribution_family: str | None = None
    parameter_priors: dict[str, Any] | None = None


@dataclass
class CausalEdge:
    """A directed edge in the causal DAG."""

    source_id: str
    target_id: str
    relation_type: str
    direction: CausalDirection
    strength: float = 1.0

    # Pyro Phase 2: structural equation metadata
    functional_form: str | None = None
    learned_params: dict[str, Any] | None = None


@dataclass(frozen=True)
class CausalAnnotation:
    """Causal annotation stored in ConceptEdge.properties['causal'].

    Frozen so it can be hashed / used as a dict key.
    """

    direction: CausalDirection
    strength: float
    functional_form: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction.value,
            "strength": self.strength,
            "functional_form": self.functional_form,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CausalAnnotation:
        return cls(
            direction=CausalDirection(data["direction"]),
            strength=data["strength"],
            functional_form=data.get("functional_form"),
        )


@dataclass
class CausalQuery:
    """A causal query — observational, interventional, or counterfactual."""

    query_type: QueryType
    target_nodes: frozenset[str]
    conditioning_nodes: frozenset[str] = field(default_factory=frozenset)

    # Phase 2+: interventions
    intervention_nodes: frozenset[str] = field(default_factory=frozenset)
    intervention_values: dict[str, Any] = field(default_factory=dict)


@dataclass
class IndependenceAssertion:
    """Result of a d-separation or conditional independence test."""

    x: frozenset[str]
    y: frozenset[str]
    z: frozenset[str]
    is_independent: bool
    method: str  # "d_separation", "chi_squared", etc.
    p_value: float | None = None
    confidence: float = 1.0


@dataclass
class CausalResult:
    """Full result of a causal query."""

    query: CausalQuery
    independences: list[IndependenceAssertion] = field(default_factory=list)
    active_paths: list[list[str]] = field(default_factory=list)

    # Phase 2+: effect estimation
    causal_effect: float | None = None
    effect_bounds: tuple[float, float] | None = None

    backend_used: str = ""
    capabilities_used: list[str] = field(default_factory=list)


@dataclass
class CreditAssignment:
    """Credit assigned to a concept for a reward signal."""

    concept_id: str
    credit: float
    path: list[str]
    method: str  # "direct", "ancestor"


# =============================================================================
# Relation → Causal Direction Mapping
# =============================================================================

RELATION_CAUSAL_DIRECTION: dict[str, tuple[CausalDirection, float]] = {
    "requires": (CausalDirection.FORWARD, 0.9),
    "implements": (CausalDirection.REVERSE, 0.85),
    "refines": (CausalDirection.REVERSE, 0.8),
    "part_of": (CausalDirection.REVERSE, 0.8),
    "uses": (CausalDirection.FORWARD, 0.75),
    "supports": (CausalDirection.FORWARD, 0.7),
    "challenges": (CausalDirection.FORWARD, 0.7),
    "contradicts": (CausalDirection.BIDIRECTIONAL, 0.7),
    "similar_to": (CausalDirection.NONE, 0.3),
    "alternative_to": (CausalDirection.NONE, 0.3),
}
