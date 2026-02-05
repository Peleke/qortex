"""FlatYAMLTarget -- serialize rules to flat YAML string."""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from qortex.core.models import Rule
from qortex.projectors.models import EnrichedRule


@dataclass
class FlatYAMLTarget:
    """Serialize rules to flat YAML format.

    Output is a YAML string with a list of rule dicts.
    Implements the ProjectionTarget[str] protocol.
    """

    include_enrichment: bool = True

    def serialize(self, rules: list[EnrichedRule] | list[Rule]) -> str:
        """Serialize rules to YAML string."""
        entries = [self._rule_to_dict(r) for r in rules]
        return yaml.dump(
            {"rules": entries},
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
        )

    def _rule_to_dict(self, rule: EnrichedRule | Rule) -> dict:
        """Convert a rule (enriched or plain) to a dict."""
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
            if self.include_enrichment and rule.enrichment:
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
