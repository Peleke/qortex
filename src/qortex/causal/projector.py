"""CausalRuleProjector — derives independence rules from causal DAGs.

Implements the ``ProjectionSource`` protocol (same as ``FlatRuleSource``).
Builds a CausalDAG per domain, runs d-separation, converts
``IndependenceAssertion``s to ``Rule`` objects with ``derivation="causal"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qortex.core.models import Rule
from qortex.projectors.models import ProjectionFilter

from .dag import CausalDAG
from .dsep import DSeparationEngine
from .types import RELATION_CAUSAL_DIRECTION, CausalDirection, IndependenceAssertion

if TYPE_CHECKING:
    from qortex.core.backend import GraphBackend


# =============================================================================
# Rule text templates
# =============================================================================

_INDEPENDENCE_TEMPLATE = (
    "{x} and {y} are conditionally independent given {z} (d-separated in the causal graph)"
)
_INDEPENDENCE_UNCONDITIONAL_TEMPLATE = (
    "{x} and {y} are unconditionally independent (d-separated in the causal graph)"
)
_DEPENDENCY_TEMPLATE = "{x} and {y} are causally dependent (d-connected in the causal graph)"


@dataclass
class CausalRuleProjector:
    """Derives causal independence/dependency rules from the knowledge graph.

    Implements the ``ProjectionSource`` protocol:
        ``derive(domains, filters) -> list[Rule]``

    Usage::

        from qortex.projectors.projection import Projection
        from qortex.projectors.enrichers.passthrough import PassthroughEnricher

        proj = Projection(
            source=CausalRuleProjector(backend=backend),
            enricher=PassthroughEnricher(),
            target=some_target,
        )
        result = proj.project(domains=["my_domain"])
    """

    backend: GraphBackend
    max_conditioning_size: int = 3
    include_dependencies: bool = False
    relation_mapping: dict[str, tuple[CausalDirection, float]] = field(
        default_factory=lambda: dict(RELATION_CAUSAL_DIRECTION)
    )
    _seen_pairs: set[tuple[str, str]] = field(default_factory=set, repr=False)

    def derive(
        self,
        domains: list[str] | None = None,
        filters: ProjectionFilter | None = None,
    ) -> list[Rule]:
        """Derive causal rules from the graph, optionally filtered."""
        self._seen_pairs = set()
        filt = filters or ProjectionFilter()

        # Skip if filter explicitly excludes causal rules
        if filt.derivation not in ("causal", "all"):
            return []

        target_domains = domains or [d.name for d in self.backend.list_domains()]
        rules: list[Rule] = []

        for domain in target_domains:
            dag = CausalDAG.from_backend(
                self.backend,
                domain=domain,
                relation_mapping=self.relation_mapping,
            )
            if len(dag.node_ids) < 2:
                continue

            engine = DSeparationEngine(dag=dag)
            assertions = engine.find_all_d_separations(
                max_conditioning_size=self.max_conditioning_size,
            )

            for assertion in assertions:
                rule = self._assertion_to_rule(assertion, domain)
                if rule is not None and self._passes_filter(rule, filt):
                    rules.append(rule)

        return rules

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _assertion_to_rule(
        self,
        assertion: IndependenceAssertion,
        domain: str,
    ) -> Rule | None:
        """Convert an IndependenceAssertion to a Rule, deduplicating by (x, y) pair."""
        # Normalize pair for dedup: sorted tuple of frozen sets
        pair_key = (
            ",".join(sorted(assertion.x)),
            ",".join(sorted(assertion.y)),
        )
        if pair_key in self._seen_pairs:
            return None
        self._seen_pairs.add(pair_key)

        x_str = ", ".join(sorted(assertion.x))
        y_str = ", ".join(sorted(assertion.y))
        source_concepts = sorted(assertion.x | assertion.y)

        if assertion.is_independent:
            if assertion.z:
                z_str = ", ".join(sorted(assertion.z))
                text = _INDEPENDENCE_TEMPLATE.format(x=x_str, y=y_str, z=z_str)
            else:
                text = _INDEPENDENCE_UNCONDITIONAL_TEMPLATE.format(x=x_str, y=y_str)
        else:
            if not self.include_dependencies:
                return None
            text = _DEPENDENCY_TEMPLATE.format(x=x_str, y=y_str)

        rule_id = f"causal:{pair_key[0]}<>{pair_key[1]}"
        return Rule(
            id=rule_id,
            text=text,
            domain=domain,
            derivation="causal",
            source_concepts=source_concepts,
            confidence=assertion.confidence,
            category="causal_independence" if assertion.is_independent else "causal_dependency",
            metadata={
                "conditioning_set": sorted(assertion.z) if assertion.z else [],
                "method": assertion.method,
            },
        )

    @staticmethod
    def _passes_filter(rule: Rule, filt: ProjectionFilter) -> bool:
        """Check if a rule passes the projection filter."""
        if filt.min_confidence > 0 and rule.confidence < filt.min_confidence:
            return False
        if filt.domains and rule.domain not in filt.domains:
            return False
        return not (filt.categories and rule.category not in filt.categories)
