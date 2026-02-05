"""Projection protocols — the three concerns of the projection pipeline.

ProjectionSource: derives rules from the KG
Enricher: adds context/antipattern/rationale to rules
ProjectionTarget: serializes enriched rules to a target format
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from qortex.core.models import Rule
from qortex.projectors.models import EnrichedRule, ProjectionFilter

T = TypeVar("T")


@runtime_checkable
class ProjectionSource(Protocol):
    """Derives rules from the knowledge graph."""

    def derive(
        self,
        domains: list[str] | None = None,
        filters: ProjectionFilter | None = None,
    ) -> list[Rule]:
        """Derive rules from the graph, optionally filtered."""
        ...


@runtime_checkable
class Enricher(Protocol):
    """Adds context, antipatterns, rationale to rules."""

    def enrich(self, rules: list[Rule]) -> list[EnrichedRule]:
        """Enrich a list of rules. Returns EnrichedRule instances."""
        ...


@runtime_checkable
class ProjectionTarget(Protocol[T]):
    """Serializes rules to a target format."""

    def serialize(self, rules: list[EnrichedRule] | list[Rule]) -> T:
        """Serialize rules to the target format."""
        ...
