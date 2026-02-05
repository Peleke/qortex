"""Projection data models — filters, enrichments, enriched rules."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from qortex.core.models import RelationType, Rule


@dataclass
class ProjectionFilter:
    """Filter criteria for rule projection."""

    domains: list[str] | None = None
    categories: list[str] | None = None
    min_confidence: float = 0.0
    derivation: Literal["explicit", "derived", "all"] = "all"
    relation_types: list[RelationType] | None = None


@dataclass
class RuleEnrichment:
    """Enrichment metadata attached to a rule by an Enricher."""

    context: str  # "When writing callbacks for map/filter..."
    antipattern: str  # "e.g., modifying external array inside .map()"
    rationale: str  # "Mixing pure functions with mutable state leads to..."
    tags: list[str]  # ["fp", "immutability", "antipattern"]
    enrichment_version: int = 1
    enriched_at: datetime | None = None
    enrichment_source: str | None = None  # "anthropic", "template"
    source_contexts: list[str] = field(default_factory=list)


@dataclass
class EnrichedRule:
    """A rule with optional enrichment metadata.

    Not a subclass of Rule — composition over inheritance.
    Holds the original rule + enrichment separately.
    """

    rule: Rule
    enrichment: RuleEnrichment | None = None

    # Delegate common accessors
    @property
    def id(self) -> str:
        return self.rule.id

    @property
    def text(self) -> str:
        return self.rule.text

    @property
    def domain(self) -> str:
        return self.rule.domain

    @property
    def confidence(self) -> float:
        return self.rule.confidence

    @property
    def category(self) -> str | None:
        return self.rule.category

    @property
    def derivation(self) -> str:
        return self.rule.derivation

    @property
    def source_concepts(self) -> list[str]:
        return self.rule.source_concepts

    @property
    def relevance(self) -> float:
        return self.rule.relevance
