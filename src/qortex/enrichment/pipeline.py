"""Enrichment pipeline — orchestrates batch enrichment of rules."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from qortex.core.models import Rule
from qortex.enrichment.base import EnrichmentBackend
from qortex.projectors.models import EnrichedRule, RuleEnrichment

logger = logging.getLogger(__name__)


@dataclass
class EnrichmentStats:
    """Track enrichment outcomes."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0


class TemplateEnrichmentFallback:
    """Mechanical fallback (no LLM). Produces basic enrichments from rule text."""

    def enrich_batch(
        self,
        rules: list[Rule],
        domain: str,
    ) -> list[RuleEnrichment]:
        return [self._enrich_one(rule) for rule in rules]

    def re_enrich(
        self,
        rule: Rule,
        existing: RuleEnrichment,
        new_context: str,
    ) -> RuleEnrichment:
        return RuleEnrichment(
            context=existing.context,
            antipattern=existing.antipattern,
            rationale=f"{existing.rationale} Additionally: {new_context}",
            tags=existing.tags,
            enrichment_version=existing.enrichment_version + 1,
            enriched_at=datetime.now(timezone.utc),
            enrichment_source="template",
            source_contexts=[*existing.source_contexts, new_context],
        )

    def _enrich_one(self, rule: Rule) -> RuleEnrichment:
        # Derive basic tags from rule text
        tags = [rule.domain]
        if rule.category:
            tags.append(rule.category)
        if rule.derivation == "derived":
            tags.append("derived")

        return RuleEnrichment(
            context=f"When working in the {rule.domain} domain",
            antipattern="Violating this rule",
            rationale=rule.text,
            tags=tags,
            enrichment_version=1,
            enriched_at=datetime.now(timezone.utc),
            enrichment_source="template",
        )


@dataclass
class EnrichmentPipeline:
    """Orchestrates enrichment of rules via a backend with fallback.

    Usage:
        pipeline = EnrichmentPipeline(backend=AnthropicEnrichmentBackend())
        enriched = pipeline.enrich(rules, domain="error_handling")
    """

    backend: EnrichmentBackend | None = None
    fallback: TemplateEnrichmentFallback = field(
        default_factory=TemplateEnrichmentFallback
    )
    stats: EnrichmentStats = field(default_factory=EnrichmentStats)

    def enrich(
        self,
        rules: list[Rule],
        domain: str,
    ) -> list[EnrichedRule]:
        """Enrich rules via backend, falling back to template on failure."""
        self.stats = EnrichmentStats(total=len(rules))

        if not rules:
            return []

        enrichments: list[RuleEnrichment] = []

        if self.backend:
            try:
                enrichments = self.backend.enrich_batch(rules, domain)
                self.stats.succeeded = len(enrichments)
            except Exception:
                logger.warning(
                    "Enrichment backend failed, falling back to template",
                    exc_info=True,
                )
                enrichments = self.fallback.enrich_batch(rules, domain)
                self.stats.failed = len(rules)
        else:
            enrichments = self.fallback.enrich_batch(rules, domain)
            self.stats.succeeded = len(enrichments)

        # Pair rules with enrichments
        result: list[EnrichedRule] = []
        for i, rule in enumerate(rules):
            enrichment = enrichments[i] if i < len(enrichments) else None
            result.append(EnrichedRule(rule=rule, enrichment=enrichment))

        return result
