"""FlatJSONTarget -- serialize rules to JSON string."""

from __future__ import annotations

import json
from dataclasses import dataclass

from qortex.core.models import Rule
from qortex.projectors.models import EnrichedRule


@dataclass
class FlatJSONTarget:
    """Serialize rules to JSON format.

    Output is a JSON string with a list of rule dicts.
    Implements the ProjectionTarget[str] protocol.
    """

    include_enrichment: bool = True
    indent: int = 2

    def serialize(self, rules: list[EnrichedRule] | list[Rule]) -> str:
        """Serialize rules to JSON string."""
        entries = [self._rule_to_dict(r) for r in rules]
        return json.dumps({"rules": entries}, indent=self.indent, ensure_ascii=False)

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
