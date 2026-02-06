"""Projectors — transform KG knowledge into consumable formats.

Core abstractions:
- ProjectionSource: derives rules from the graph
- Enricher: adds context/antipattern/rationale
- ProjectionTarget: serializes to target format
- Projection: orchestrates source → enricher → target
"""

from qortex.projectors.base import Enricher, ProjectionSource, ProjectionTarget
from qortex.projectors.models import EnrichedRule, ProjectionFilter, RuleEnrichment
from qortex.projectors.projection import Projection

__all__ = [
    "Enricher",
    "EnrichedRule",
    "Projection",
    "ProjectionFilter",
    "ProjectionSource",
    "ProjectionTarget",
    "RuleEnrichment",
]
