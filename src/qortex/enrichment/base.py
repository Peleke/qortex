"""Enrichment backend protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from qortex.core.models import Rule
from qortex.projectors.models import RuleEnrichment


@runtime_checkable
class EnrichmentBackend(Protocol):
    """Backend that enriches rules with context, antipatterns, rationale."""

    def enrich_batch(
        self,
        rules: list[Rule],
        domain: str,
    ) -> list[RuleEnrichment]:
        """Enrich a batch of rules. Returns one RuleEnrichment per rule."""
        ...

    def re_enrich(
        self,
        rule: Rule,
        existing: RuleEnrichment,
        new_context: str,
    ) -> RuleEnrichment:
        """Re-enrich a rule with new context, preserving existing enrichment."""
        ...
