"""BuildlogSeedTarget -- serialize rules to buildlog seed YAML dict.

Produces a dict structure compatible with buildlog's SeedFile.from_dict().
The output includes persona metadata, version info, and rules with
enrichment fields (context, antipattern, rationale).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from qortex.core.models import Rule
from qortex.projectors.models import EnrichedRule


@dataclass
class BuildlogSeedTarget:
    """Serialize rules to buildlog seed format.

    Output is a dict that can be written to YAML and loaded by
    buildlog's SeedFile.from_dict().

    Implements the ProjectionTarget[dict] protocol.
    """

    persona_name: str = "qortex"
    persona_description: str = "Knowledge graph derived coding rules"
    version: str = "1.0.0"
    extra_metadata: dict = field(default_factory=dict)

    def serialize(self, rules: list[EnrichedRule] | list[Rule]) -> dict:
        """Serialize rules to buildlog seed dict."""
        seed_rules = [self._rule_to_seed(r) for r in rules]

        return {
            "persona": {
                "name": self.persona_name,
                "description": self.persona_description,
            },
            "version": self.version,
            "metadata": {
                "source": "qortex",
                "rule_count": len(seed_rules),
                **self.extra_metadata,
            },
            "rules": seed_rules,
        }

    def _rule_to_seed(self, rule: EnrichedRule | Rule) -> dict:
        """Convert a rule to a buildlog seed rule entry."""
        if isinstance(rule, EnrichedRule):
            entry = {
                "id": rule.id,
                "text": rule.text,
                "domain": rule.domain,
                "confidence": rule.confidence,
                "derivation": rule.derivation,
            }
            if rule.category:
                entry["category"] = rule.category
            if rule.enrichment:
                entry["context"] = rule.enrichment.context
                entry["antipattern"] = rule.enrichment.antipattern
                entry["rationale"] = rule.enrichment.rationale
                entry["tags"] = rule.enrichment.tags
            return entry
        else:
            entry = {
                "id": rule.id,
                "text": rule.text,
                "domain": rule.domain,
                "confidence": rule.confidence,
                "derivation": rule.derivation,
            }
            if rule.category:
                entry["category"] = rule.category
            return entry
