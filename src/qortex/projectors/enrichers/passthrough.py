"""PassthroughEnricher -- identity enricher that wraps rules without enrichment."""

from __future__ import annotations

from qortex.core.models import Rule
from qortex.projectors.models import EnrichedRule


class PassthroughEnricher:
    """Wraps rules in EnrichedRule with no enrichment metadata.

    Use when you need the Enricher protocol but don't want actual enrichment.
    Implements the Enricher protocol.
    """

    def enrich(self, rules: list[Rule]) -> list[EnrichedRule]:
        """Wrap each rule in an EnrichedRule with enrichment=None."""
        return [EnrichedRule(rule=rule, enrichment=None) for rule in rules]
