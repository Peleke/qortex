"""Shared serialization helpers for projection targets."""

from __future__ import annotations

from qortex.core.models import Rule
from qortex.projectors.models import EnrichedRule


def rule_to_dict(
    rule: EnrichedRule | Rule,
    include_enrichment: bool = True,
) -> dict:
    """Convert a rule (enriched or plain) to a serializable dict.

    Shared by FlatYAMLTarget, FlatJSONTarget, and any future targets.
    """
    if isinstance(rule, EnrichedRule):
        d = {
            "id": rule.id,
            "text": rule.text,
            "domain": rule.domain,
            "derivation": rule.derivation,
            "confidence": rule.confidence,
        }
        if rule.category:
            d["category"] = rule.category
        if rule.source_concepts:
            d["source_concepts"] = rule.source_concepts
        if include_enrichment and rule.enrichment:
            d["enrichment"] = {
                "context": rule.enrichment.context,
                "antipattern": rule.enrichment.antipattern,
                "rationale": rule.enrichment.rationale,
                "tags": rule.enrichment.tags,
            }
        return d
    else:
        d = {
            "id": rule.id,
            "text": rule.text,
            "domain": rule.domain,
            "derivation": rule.derivation,
            "confidence": rule.confidence,
        }
        if rule.category:
            d["category"] = rule.category
        if rule.source_concepts:
            d["source_concepts"] = rule.source_concepts
        return d
