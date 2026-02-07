"""Rule collection for activated concepts.

Pure function: given a set of concept IDs (from query/explore), collect all
linked rules and score them by relevance. Used by query(), explore(), and
rules() — single source of truth for "which rules apply here?".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qortex.core.backend import GraphBackend
    from qortex.core.models import Rule


def collect_rules_for_concepts(
    backend: GraphBackend,
    concept_ids: list[str],
    domains: list[str] | None = None,
    scores: dict[str, float] | None = None,
) -> list[Rule]:
    """Collect rules linked to the given concept IDs.

    Args:
        backend: GraphBackend to fetch rules from.
        concept_ids: Concept IDs whose linked rules to collect.
        domains: Optional domain filter. None = all domains.
        scores: Optional activation scores per concept_id (from PPR/vec).
            Used to set rule relevance = max score across linked concepts.

    Returns:
        List of Rule objects sorted by relevance descending. Each rule's
        relevance is the max activation score of its linked concepts.
    """
    from qortex.core.models import Rule

    if not concept_ids:
        return []

    concept_set = set(concept_ids)
    scores_map = scores or {}

    # Collect all rules from target domains
    target_domains = domains
    if target_domains is None:
        target_domains = [d.name for d in backend.list_domains()]

    seen_ids: set[str] = set()
    rules: list[Rule] = []

    for domain_name in target_domains:
        for explicit_rule in backend.get_rules(domain_name):
            if explicit_rule.id in seen_ids:
                continue

            # Check overlap: rule's concept_ids ∩ activated concept_ids
            overlap = concept_set.intersection(explicit_rule.concept_ids)
            if not overlap:
                continue

            seen_ids.add(explicit_rule.id)

            # Relevance = max activation score across linked concepts
            relevance = max(
                (scores_map.get(cid, 0.0) for cid in overlap),
                default=0.0,
            )

            rules.append(
                Rule(
                    id=explicit_rule.id,
                    text=explicit_rule.text,
                    domain=explicit_rule.domain,
                    derivation="explicit",
                    source_concepts=explicit_rule.concept_ids,
                    confidence=explicit_rule.confidence,
                    relevance=round(relevance, 4),
                    category=explicit_rule.category,
                )
            )

    # Sort by relevance descending, then by id for stability
    rules.sort(key=lambda r: (-r.relevance, r.id))
    return rules
