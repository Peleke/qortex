"""Edge rule template registry — 39 variants across 13 relation types.

Each RelationType has 3 template variants with different severity levels
and applicability contexts. Templates are used by FlatRuleSource to derive
rules from KG edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import RelationType


@dataclass(frozen=True)
class EdgeRuleTemplate:
    """A template for deriving a rule from a KG edge."""

    id: str  # "contradicts:avoidance"
    relation_type: RelationType
    template: str  # Contains {source}, {target} placeholders
    variant: str  # "avoidance", "ordering", etc.
    category: str  # "antipattern", "architectural", "general"
    severity: Literal["info", "warning", "error"]
    applicability: str  # When this variant is most relevant


# =============================================================================
# Registry: 30 templates (3 per RelationType)
# =============================================================================

EDGE_RULE_TEMPLATE_REGISTRY: list[EdgeRuleTemplate] = [
    # --- CONTRADICTS ---
    EdgeRuleTemplate(
        id="contradicts:avoidance",
        relation_type=RelationType.CONTRADICTS,
        template="When applying {source}, avoid {target}; they are fundamentally incompatible",
        variant="avoidance",
        category="antipattern",
        severity="warning",
        applicability="When one approach is chosen and the other should be explicitly avoided",
    ),
    EdgeRuleTemplate(
        id="contradicts:mutual_exclusion",
        relation_type=RelationType.CONTRADICTS,
        template="{source} and {target} are mutually exclusive; using both is a design error",
        variant="mutual_exclusion",
        category="architectural",
        severity="error",
        applicability="When both concepts cannot coexist in the same design",
    ),
    EdgeRuleTemplate(
        id="contradicts:tension",
        relation_type=RelationType.CONTRADICTS,
        template="There is a tension between {source} and {target}; be explicit about which you choose",
        variant="tension",
        category="general",
        severity="info",
        applicability="When concepts conflict but context determines the right choice",
    ),
    # --- REQUIRES ---
    EdgeRuleTemplate(
        id="requires:dependency",
        relation_type=RelationType.REQUIRES,
        template="{target} depends on {source}; omitting {source} will cause {target} to fail",
        variant="dependency",
        category="architectural",
        severity="error",
        applicability="When the dependency is hard and target literally cannot work without source",
    ),
    EdgeRuleTemplate(
        id="requires:ordering",
        relation_type=RelationType.REQUIRES,
        template="Before applying {target}, ensure {source} is in place",
        variant="ordering",
        category="architectural",
        severity="warning",
        applicability="When order of operations matters for correctness",
    ),
    EdgeRuleTemplate(
        id="requires:prerequisite",
        relation_type=RelationType.REQUIRES,
        template="{source} is a prerequisite for correctly applying {target}",
        variant="prerequisite",
        category="general",
        severity="info",
        applicability="When source is needed but the dependency is soft or conceptual",
    ),
    # --- REFINES ---
    EdgeRuleTemplate(
        id="refines:specialization",
        relation_type=RelationType.REFINES,
        template="{target} is a more specific form of {source}; prefer when context demands precision",
        variant="specialization",
        category="general",
        severity="info",
        applicability="When the refinement adds precision to a general concept",
    ),
    EdgeRuleTemplate(
        id="refines:precision",
        relation_type=RelationType.REFINES,
        template="Consider using {target} instead of {source} when higher precision is needed",
        variant="precision",
        category="architectural",
        severity="warning",
        applicability="When the general form may be too broad for the situation",
    ),
    EdgeRuleTemplate(
        id="refines:scope",
        relation_type=RelationType.REFINES,
        template="{target} narrows the scope of {source}; apply {target} when the broader concept is too general",
        variant="scope",
        category="general",
        severity="info",
        applicability="When scoping decisions affect design choices",
    ),
    # --- IMPLEMENTS ---
    EdgeRuleTemplate(
        id="implements:realization",
        relation_type=RelationType.IMPLEMENTS,
        template="{target} is a concrete way to achieve {source}",
        variant="realization",
        category="general",
        severity="info",
        applicability="When mapping abstract concepts to concrete implementations",
    ),
    EdgeRuleTemplate(
        id="implements:concrete",
        relation_type=RelationType.IMPLEMENTS,
        template="To implement {source} in practice, use {target}",
        variant="concrete",
        category="architectural",
        severity="warning",
        applicability="When practical implementation guidance is needed",
    ),
    EdgeRuleTemplate(
        id="implements:mapping",
        relation_type=RelationType.IMPLEMENTS,
        template="{target} maps the abstract concept of {source} to a concrete pattern",
        variant="mapping",
        category="general",
        severity="info",
        applicability="When documenting how abstractions become implementations",
    ),
    # --- PART_OF ---
    EdgeRuleTemplate(
        id="part_of:composition",
        relation_type=RelationType.PART_OF,
        template="{source} is a component of {target}; changes to {source} affect {target}",
        variant="composition",
        category="architectural",
        severity="info",
        applicability="When understanding compositional relationships and change propagation",
    ),
    EdgeRuleTemplate(
        id="part_of:completeness",
        relation_type=RelationType.PART_OF,
        template="When implementing {target}, ensure {source} is included",
        variant="completeness",
        category="architectural",
        severity="warning",
        applicability="When a component might be forgotten during implementation",
    ),
    EdgeRuleTemplate(
        id="part_of:decomposition",
        relation_type=RelationType.PART_OF,
        template="{target} can be decomposed into parts including {source}",
        variant="decomposition",
        category="general",
        severity="info",
        applicability="When breaking down a complex concept into manageable parts",
    ),
    # --- USES ---
    EdgeRuleTemplate(
        id="uses:coupling",
        relation_type=RelationType.USES,
        template="{source} uses {target}, creating coupling; changes to {target} may require updates to {source}",
        variant="coupling",
        category="architectural",
        severity="warning",
        applicability="When highlighting coupling risks between components",
    ),
    EdgeRuleTemplate(
        id="uses:dependency",
        relation_type=RelationType.USES,
        template="{source} depends on {target} at runtime",
        variant="dependency",
        category="general",
        severity="info",
        applicability="When documenting runtime dependencies",
    ),
    EdgeRuleTemplate(
        id="uses:interface",
        relation_type=RelationType.USES,
        template="{source} interacts with {target} through its interface; ensure the interface contract is stable",
        variant="interface",
        category="architectural",
        severity="info",
        applicability="When interface stability affects dependent components",
    ),
    # --- SIMILAR_TO ---
    EdgeRuleTemplate(
        id="similar_to:analogy",
        relation_type=RelationType.SIMILAR_TO,
        template="{source} and {target} are analogous; patterns from one may transfer to the other",
        variant="analogy",
        category="general",
        severity="info",
        applicability="When knowledge transfer between similar concepts is valuable",
    ),
    EdgeRuleTemplate(
        id="similar_to:transferability",
        relation_type=RelationType.SIMILAR_TO,
        template="Lessons learned from {source} likely apply to {target} as well",
        variant="transferability",
        category="general",
        severity="info",
        applicability="When cross-pollinating insights between related areas",
    ),
    EdgeRuleTemplate(
        id="similar_to:distinction",
        relation_type=RelationType.SIMILAR_TO,
        template="{source} and {target} appear similar but have important differences; don't conflate them",
        variant="distinction",
        category="architectural",
        severity="warning",
        applicability="When similar concepts have subtle but important differences",
    ),
    # --- ALTERNATIVE_TO ---
    EdgeRuleTemplate(
        id="alternative_to:selection",
        relation_type=RelationType.ALTERNATIVE_TO,
        template="{source} and {target} are alternatives; choose based on context",
        variant="selection",
        category="general",
        severity="info",
        applicability="When a neutral choice between alternatives is needed",
    ),
    EdgeRuleTemplate(
        id="alternative_to:tradeoff",
        relation_type=RelationType.ALTERNATIVE_TO,
        template="Choosing {source} over {target} involves tradeoffs; document your rationale",
        variant="tradeoff",
        category="architectural",
        severity="warning",
        applicability="When the choice between alternatives has significant consequences",
    ),
    EdgeRuleTemplate(
        id="alternative_to:substitution",
        relation_type=RelationType.ALTERNATIVE_TO,
        template="{source} can substitute for {target} in most contexts",
        variant="substitution",
        category="general",
        severity="info",
        applicability="When alternatives are largely interchangeable",
    ),
    # --- SUPPORTS ---
    EdgeRuleTemplate(
        id="supports:evidence",
        relation_type=RelationType.SUPPORTS,
        template="{source} provides evidence supporting the use of {target}",
        variant="evidence",
        category="general",
        severity="info",
        applicability="When documenting evidence chains for design decisions",
    ),
    EdgeRuleTemplate(
        id="supports:reinforcement",
        relation_type=RelationType.SUPPORTS,
        template="{source} reinforces {target}; they work well together",
        variant="reinforcement",
        category="general",
        severity="info",
        applicability="When concepts are synergistic",
    ),
    EdgeRuleTemplate(
        id="supports:validation",
        relation_type=RelationType.SUPPORTS,
        template="{source} validates the approach taken by {target}; use {source} to verify correctness",
        variant="validation",
        category="architectural",
        severity="warning",
        applicability="When one concept can be used to validate another",
    ),
    # --- CHALLENGES ---
    EdgeRuleTemplate(
        id="challenges:counter_evidence",
        relation_type=RelationType.CHALLENGES,
        template="{source} provides counter-evidence to {target}; consider before committing",
        variant="counter_evidence",
        category="general",
        severity="warning",
        applicability="When there is evidence against a particular approach",
    ),
    EdgeRuleTemplate(
        id="challenges:limitation",
        relation_type=RelationType.CHALLENGES,
        template="{source} reveals a limitation of {target}; account for this in your design",
        variant="limitation",
        category="general",
        severity="info",
        applicability="When one concept exposes weaknesses in another",
    ),
    EdgeRuleTemplate(
        id="challenges:caveat",
        relation_type=RelationType.CHALLENGES,
        template="{source} introduces a caveat to {target}; apply {target} with this caveat in mind",
        variant="caveat",
        category="architectural",
        severity="warning",
        applicability="When caveats affect how a concept should be applied",
    ),
    # --- BELONGS_TO (database-derived: ownership FK) ---
    EdgeRuleTemplate(
        id="belongs_to:ownership",
        relation_type=RelationType.BELONGS_TO,
        template="{source} belongs to {target}; access control should respect this ownership",
        variant="ownership",
        category="architectural",
        severity="warning",
        applicability="When ownership determines access control or data scoping",
    ),
    EdgeRuleTemplate(
        id="belongs_to:scoping",
        relation_type=RelationType.BELONGS_TO,
        template="{source} is scoped to {target}; queries should filter by this ownership",
        variant="scoping",
        category="architectural",
        severity="info",
        applicability="When data should be filtered by owner in queries",
    ),
    EdgeRuleTemplate(
        id="belongs_to:lifecycle",
        relation_type=RelationType.BELONGS_TO,
        template="{source} exists in the context of {target}; consider lifecycle implications",
        variant="lifecycle",
        category="general",
        severity="info",
        applicability="When the owned entity's lifecycle depends on the owner",
    ),
    # --- INSTANCE_OF (database-derived: template/catalog FK) ---
    EdgeRuleTemplate(
        id="instance_of:instantiation",
        relation_type=RelationType.INSTANCE_OF,
        template="{source} is an instance of {target}; it inherits the template's properties",
        variant="instantiation",
        category="general",
        severity="info",
        applicability="When instances derive behavior from templates or catalogs",
    ),
    EdgeRuleTemplate(
        id="instance_of:conformance",
        relation_type=RelationType.INSTANCE_OF,
        template="{source} should conform to the contract defined by {target}",
        variant="conformance",
        category="architectural",
        severity="warning",
        applicability="When instances must match template specifications",
    ),
    EdgeRuleTemplate(
        id="instance_of:variation",
        relation_type=RelationType.INSTANCE_OF,
        template="{source} is a variant of {target}; customizations should be documented",
        variant="variation",
        category="general",
        severity="info",
        applicability="When instances may diverge from templates",
    ),
    # --- CONTAINS (database-derived: parent-child containment) ---
    EdgeRuleTemplate(
        id="contains:aggregation",
        relation_type=RelationType.CONTAINS,
        template="{source} contains {target}; operations on {source} may cascade to {target}",
        variant="aggregation",
        category="architectural",
        severity="warning",
        applicability="When parent operations affect contained children",
    ),
    EdgeRuleTemplate(
        id="contains:boundary",
        relation_type=RelationType.CONTAINS,
        template="{target} lives within the boundary of {source}; it should not be accessed independently",
        variant="boundary",
        category="architectural",
        severity="info",
        applicability="When containment implies aggregate root boundaries",
    ),
    EdgeRuleTemplate(
        id="contains:enumeration",
        relation_type=RelationType.CONTAINS,
        template="{source} has {target} as a component; ensure all components are accounted for",
        variant="enumeration",
        category="general",
        severity="info",
        applicability="When listing all parts of a container",
    ),
]


# =============================================================================
# Lookup functions
# =============================================================================


def get_templates(relation_type: RelationType) -> list[EdgeRuleTemplate]:
    """Get all template variants for a relation type."""
    return [t for t in EDGE_RULE_TEMPLATE_REGISTRY if t.relation_type == relation_type]


def get_default_template(relation_type: RelationType) -> EdgeRuleTemplate:
    """Get the default (first) template for a relation type."""
    templates = get_templates(relation_type)
    if not templates:
        raise ValueError(f"No templates for relation type: {relation_type}")
    return templates[0]


def select_template(
    relation_type: RelationType,
    category_hint: str | None = None,
) -> EdgeRuleTemplate:
    """Select the best template for a relation type.

    If category_hint matches a template's category, prefer that.
    Otherwise return the default (first) variant.
    """
    templates = get_templates(relation_type)
    if not templates:
        raise ValueError(f"No templates for relation type: {relation_type}")

    if category_hint:
        for t in templates:
            if t.category == category_hint:
                return t

    return templates[0]
