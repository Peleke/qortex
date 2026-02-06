"""TemplateEnricher -- mechanical enrichment without LLM calls."""

from __future__ import annotations

from dataclasses import dataclass, field

from qortex.core.models import Rule
from qortex.enrichment.pipeline import TemplateEnrichmentFallback
from qortex.projectors.models import EnrichedRule


@dataclass
class TemplateEnricher:
    """Enriches rules using mechanical templates (no LLM).

    Wraps TemplateEnrichmentFallback from the enrichment package.
    Implements the Enricher protocol.
    """

    domain: str = "general"
    _fallback: TemplateEnrichmentFallback = field(
        default_factory=TemplateEnrichmentFallback, repr=False
    )

    def enrich(self, rules: list[Rule]) -> list[EnrichedRule]:
        """Enrich rules using template-based mechanical enrichment."""
        if not rules:
            return []

        enrichments = self._fallback.enrich_batch(rules, self.domain)

        return [
            EnrichedRule(rule=rule, enrichment=enrichment)
            for rule, enrichment in zip(rules, enrichments)
        ]
