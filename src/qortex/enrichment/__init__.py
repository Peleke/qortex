"""Enrichment — add context, antipatterns, rationale to rules via LLM or templates."""

from qortex.enrichment.base import EnrichmentBackend
from qortex.enrichment.pipeline import EnrichmentPipeline

__all__ = [
    "EnrichmentBackend",
    "EnrichmentPipeline",
]
