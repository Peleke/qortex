"""LLMEnricher -- enriches rules via the EnrichmentPipeline (LLM backend)."""

from __future__ import annotations

from dataclasses import dataclass, field

from qortex.core.models import Rule
from qortex.enrichment.pipeline import EnrichmentPipeline
from qortex.projectors.models import EnrichedRule


@dataclass
class LLMEnricher:
    """Enriches rules via the EnrichmentPipeline.

    Wraps EnrichmentPipeline which handles LLM backend + template fallback.
    Implements the Enricher protocol.
    """

    pipeline: EnrichmentPipeline = field(default_factory=EnrichmentPipeline)
    domain: str = "general"

    def enrich(self, rules: list[Rule]) -> list[EnrichedRule]:
        """Enrich rules via the enrichment pipeline."""
        if not rules:
            return []

        # Infer domain from first rule if not set explicitly
        effective_domain = rules[0].domain
        return self.pipeline.enrich(rules, effective_domain)
