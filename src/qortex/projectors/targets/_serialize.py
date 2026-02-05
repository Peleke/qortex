"""Shared serialization helpers for projection targets.

Contains the universal rule set schema that any consumer can use.
BuildlogSeedTarget, FlatYAMLTarget, etc. are thin wrappers over these.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from qortex.core.models import Rule
from qortex.projectors.models import EnrichedRule


def rule_to_dict(
    rule: EnrichedRule | Rule,
    include_enrichment: bool = True,
) -> dict:
    """Convert a rule (enriched or plain) to a flat serializable dict.

    Used by FlatYAMLTarget and FlatJSONTarget for simple list-of-rules output.
    """
    d: dict = {
        "id": rule.id,
        "text": rule.text,
        "domain": rule.domain,
        "derivation": rule.derivation,
        "confidence": rule.confidence,
    }
    if rule.category:
        d["category"] = rule.category
    if rule.source_concepts:
        d["source_concepts"] = rule.source_concepts

    # Include metadata if present (template info for derived rules)
    r = rule.rule if isinstance(rule, EnrichedRule) else rule
    if r.metadata:
        d["metadata"] = r.metadata

    if (
        include_enrichment
        and isinstance(rule, EnrichedRule)
        and rule.enrichment
    ):
        d["enrichment"] = {
            "context": rule.enrichment.context,
            "antipattern": rule.enrichment.antipattern,
            "rationale": rule.enrichment.rationale,
            "tags": rule.enrichment.tags,
        }

    return d


def _build_provenance(rule: EnrichedRule | Rule) -> dict[str, Any]:
    """Build the provenance block for a rule in the universal schema."""
    r = rule.rule if isinstance(rule, EnrichedRule) else rule
    meta = r.metadata

    prov: dict[str, Any] = {
        "id": rule.id,
        "domain": rule.domain,
        "derivation": rule.derivation,
        "source_concepts": list(rule.source_concepts),
        "confidence": rule.confidence,
        "relevance": rule.relevance,
        # Template fields -- null for explicit rules, populated for derived
        "relation_type": meta.get("relation_type"),
        "template_id": meta.get("template_id"),
        "template_variant": meta.get("template_variant"),
        "template_severity": meta.get("template_severity"),
    }

    return prov


def _rule_to_seed_entry(
    rule: EnrichedRule | Rule,
    graph_version: str | None = None,
) -> dict[str, Any]:
    """Convert a rule to a universal seed entry (the shared contract).

    Output shape per rule:
        rule: <text>
        category: <string>
        context: ...         (if enriched)
        antipattern: ...     (if enriched)
        rationale: ...       (if enriched)
        tags: [...]          (if enriched)
        provenance:
            id, domain, derivation, source_concepts, confidence, relevance,
            relation_type, template_id, template_variant, template_severity,
            graph_version
    """
    entry: dict[str, Any] = {
        "rule": rule.text,
        "category": rule.category or rule.domain,
    }

    # Enrichment fields at rule level (not nested)
    if isinstance(rule, EnrichedRule) and rule.enrichment:
        entry["context"] = rule.enrichment.context
        entry["antipattern"] = rule.enrichment.antipattern
        entry["rationale"] = rule.enrichment.rationale
        entry["tags"] = rule.enrichment.tags

    # Provenance block
    prov = _build_provenance(rule)
    if graph_version is not None:
        prov["graph_version"] = graph_version
    entry["provenance"] = prov

    return entry


def serialize_ruleset(
    rules: list[EnrichedRule] | list[Rule],
    *,
    persona: str,
    version: int = 1,
    source: str = "qortex",
    source_version: str = "0.1.0",
    graph_version: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Serialize rules to the universal rule set schema.

    This is the canonical output format that any consumer can ingest.
    Buildlog, MCP servers, agent frameworks -- all use this same shape.

    Args:
        rules: List of rules (enriched or plain).
        persona: Flat string identifier (e.g. "qortex_implementation_hiding").
        version: Schema version as int (consumers compare numerically).
        source: Origin system identifier.
        source_version: Version of the source system.
        graph_version: ISO timestamp of the graph state used for projection.
        extra_metadata: Additional metadata to include.
    """
    projected_at = datetime.now(timezone.utc).isoformat()

    seed_rules = [
        _rule_to_seed_entry(r, graph_version=graph_version)
        for r in rules
    ]

    metadata: dict[str, Any] = {
        "source": source,
        "source_version": source_version,
        "projected_at": projected_at,
        "rule_count": len(seed_rules),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        "persona": persona,
        "version": version,
        "rules": seed_rules,
        "metadata": metadata,
    }
