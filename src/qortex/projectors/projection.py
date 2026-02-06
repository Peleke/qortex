"""Projection orchestrator — composes Source → Enricher → Target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from qortex.projectors.base import Enricher, ProjectionSource, ProjectionTarget
from qortex.projectors.models import ProjectionFilter

T = TypeVar("T")


@dataclass
class Projection(Generic[T]):
    """Compose source + enricher + target into a single projection pipeline.

    Usage:
        projection = Projection(
            source=FlatRuleSource(backend),
            enricher=LLMEnricher(client),  # or None
            target=BuildlogSeedTarget(),
        )
        result = projection.project(domains=["error_handling"])
    """

    source: ProjectionSource
    target: ProjectionTarget[T]
    enricher: Enricher | None = None

    def project(
        self,
        domains: list[str] | None = None,
        filters: ProjectionFilter | None = None,
    ) -> T:
        """Run the full projection pipeline."""
        rules = self.source.derive(domains=domains, filters=filters)
        if self.enricher:
            enriched = self.enricher.enrich(rules)
            return self.target.serialize(enriched)
        return self.target.serialize(rules)
