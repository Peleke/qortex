"""Ingest buildlog emission artifacts into the qortex knowledge graph.

Reads processed (and optionally pending) emission artifacts from buildlog's
~/.buildlog/emissions/ directory, aggregates concepts and edges, deduplicates,
and produces an IngestionManifest that can be loaded into Memgraph.

Emission types handled:
- mistake_manifest: mistake concept nodes + challenge/support edges
- reward_signal: reward concept nodes
- session_summary: session concept nodes + uses edges
- learned_rules: converted to ExplicitRule objects

No LLM calls required — the data is already structured.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qortex.core.models import (
    ConceptEdge,
    ConceptNode,
    ExplicitRule,
    IngestionManifest,
    RelationType,
    SourceMetadata,
)

# Default emission directories
DEFAULT_EMISSIONS_DIR = Path("~/.buildlog/emissions").expanduser()

# Valid relation type values for mapping
_VALID_RELATIONS = {r.value for r in RelationType}


@dataclass
class AggregationResult:
    """Result of aggregating emission artifacts."""

    concepts: dict[str, ConceptNode] = field(default_factory=dict)  # keyed by id
    edges: list[ConceptEdge] = field(default_factory=list)
    rules: list[ExplicitRule] = field(default_factory=list)
    seen_edge_pairs: set[tuple[str, str, str]] = field(default_factory=set)

    # Stats
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0
    by_type: dict[str, int] = field(default_factory=lambda: {
        "mistake_manifest": 0,
        "reward_signal": 0,
        "session_summary": 0,
        "learned_rules": 0,
        "unknown": 0,
    })


def _classify_artifact(filename: str) -> str:
    """Classify an emission artifact by its filename prefix."""
    for prefix in ("mistake_manifest", "reward_signal", "session_summary", "learned_rules"):
        if filename.startswith(prefix):
            return prefix
    return "unknown"


def _map_relation_type(raw: str) -> RelationType | None:
    """Map a raw relation type string to RelationType enum, or None if invalid."""
    if raw in _VALID_RELATIONS:
        return RelationType(raw)
    return None


def _extract_concept(raw: dict[str, Any], artifact_type: str) -> ConceptNode | None:
    """Convert an emission concept dict to a ConceptNode."""
    name = raw.get("name", "")
    if not name:
        return None

    # Use concept name as both id and name; prefix with domain for uniqueness
    domain = raw.get("domain", "buildlog")
    source_id = raw.get("source_id", f"buildlog:{artifact_type}")
    properties = raw.get("properties", {})

    # Build a human-readable description from properties
    description_parts = []
    if artifact_type == "mistake_manifest":
        error_class = properties.get("error_class", "unknown")
        desc = properties.get("description", "")
        description_parts.append(f"Mistake ({error_class})")
        if desc:
            description_parts.append(desc)
    elif artifact_type == "reward_signal":
        outcome = properties.get("outcome", "unknown")
        value = properties.get("reward_value", 0)
        description_parts.append(f"Reward: {outcome} (value={value})")
    elif artifact_type == "session_summary":
        duration = properties.get("duration_minutes", 0)
        mistakes = properties.get("mistakes_logged", 0)
        outcome = properties.get("outcome", "unknown")
        description_parts.append(f"Session: {duration:.0f}min, {mistakes} mistakes, {outcome}")
    else:
        description_parts.append(name)

    description = " — ".join(description_parts) if description_parts else name

    return ConceptNode(
        id=name,
        name=name,
        description=description,
        domain=domain,
        source_id=source_id,
        properties=properties,
    )


def _extract_edges(raw_edges: list[dict], artifact_type: str) -> list[tuple[str, str, str, float, dict]]:
    """Extract edge tuples from emission edge data.

    Returns list of (source_id, target_id, relation_type, confidence, properties).
    """
    results = []
    for edge in raw_edges:
        source_id = edge.get("source_id", "")
        target_id = edge.get("target_id", "")
        rel_type_str = edge.get("relation_type", "")
        confidence = edge.get("confidence", 1.0)
        properties = edge.get("properties", {})

        if not source_id or not target_id or not rel_type_str:
            continue

        results.append((source_id, target_id, rel_type_str, confidence, properties))
    return results


def _extract_learned_rules(data: dict[str, Any]) -> list[ExplicitRule]:
    """Extract ExplicitRule objects from a learned_rules artifact."""
    rules = []
    for raw_rule in data.get("rules", []):
        prov = raw_rule.get("provenance", {})
        rule_id = prov.get("id", f"learned:{len(rules)}")
        text = raw_rule.get("rule", "")
        if not text:
            continue

        rules.append(ExplicitRule(
            id=rule_id,
            text=text,
            domain=prov.get("domain", "buildlog"),
            source_id=f"buildlog:learned_rules",
            category=raw_rule.get("category"),
            confidence=prov.get("confidence", 0.7),
        ))
    return rules


def aggregate_emissions(
    emissions_dir: Path = DEFAULT_EMISSIONS_DIR,
    include_pending: bool = False,
    include_processed: bool = True,
) -> AggregationResult:
    """Aggregate emission artifacts into concepts, edges, and rules.

    Args:
        emissions_dir: Root emissions directory (contains pending/, processed/).
        include_pending: Whether to include pending/ artifacts.
        include_processed: Whether to include processed/ artifacts.

    Returns:
        AggregationResult with deduplicated concepts, edges, and rules.
    """
    result = AggregationResult()

    dirs_to_scan: list[Path] = []
    if include_processed:
        dirs_to_scan.append(emissions_dir / "processed")
    if include_pending:
        dirs_to_scan.append(emissions_dir / "pending")

    for scan_dir in dirs_to_scan:
        if not scan_dir.exists():
            continue

        for artifact_path in sorted(scan_dir.glob("*.json")):
            artifact_type = _classify_artifact(artifact_path.name)

            try:
                data = json.loads(artifact_path.read_text())
            except (json.JSONDecodeError, OSError):
                result.files_failed += 1
                continue

            result.files_processed += 1
            result.by_type[artifact_type] = result.by_type.get(artifact_type, 0) + 1

            if artifact_type == "learned_rules":
                # Different format — extract rules directly
                result.rules.extend(_extract_learned_rules(data))
                continue

            # Standard emission format: concepts + edges
            for raw_concept in data.get("concepts", []):
                concept = _extract_concept(raw_concept, artifact_type)
                if concept and concept.id not in result.concepts:
                    result.concepts[concept.id] = concept

            for source_id, target_id, rel_str, confidence, props in _extract_edges(data.get("edges", []), artifact_type):
                rel_type = _map_relation_type(rel_str)
                if rel_type is None:
                    continue

                # Dedup by (source, target, relation)
                edge_key = (source_id, target_id, rel_str)
                if edge_key in result.seen_edge_pairs:
                    continue
                result.seen_edge_pairs.add(edge_key)

                # Ensure both endpoints exist as concepts (create stubs if needed)
                for node_id in (source_id, target_id):
                    if node_id not in result.concepts:
                        result.concepts[node_id] = ConceptNode(
                            id=node_id,
                            name=node_id,
                            description=f"Reference node from {artifact_type}",
                            domain="buildlog",
                            source_id=f"buildlog:{artifact_type}",
                        )

                result.edges.append(ConceptEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=rel_type,
                    confidence=confidence,
                    properties=props,
                ))

    return result


def build_manifest(
    result: AggregationResult,
    domain: str = "buildlog",
) -> IngestionManifest:
    """Build an IngestionManifest from aggregated emission data.

    Args:
        result: Aggregated emission data.
        domain: Domain name for the manifest.

    Returns:
        IngestionManifest ready for graph ingestion.
    """
    source = SourceMetadata(
        id=f"buildlog:emissions:{datetime.now(UTC).strftime('%Y%m%d')}",
        name="buildlog_emissions",
        source_type="text",
        path_or_url=str(DEFAULT_EMISSIONS_DIR),
        chunk_count=result.files_processed,
        concept_count=len(result.concepts),
        rule_count=len(result.rules),
    )

    return IngestionManifest(
        source=source,
        domain=domain,
        concepts=list(result.concepts.values()),
        edges=result.edges,
        rules=result.rules,
    )


# =============================================================================
# Gauntlet Bridge: cross-domain edges from buildlog rules to pattern domains
# =============================================================================

DEFAULT_BUILDLOG_DB = Path("~/.buildlog/buildlog.db").expanduser()


def bridge_gauntlet_rules(
    db_path: Path = DEFAULT_BUILDLOG_DB,
) -> tuple[list[ConceptNode], list[ConceptEdge]]:
    """Read gauntlet rules from buildlog DB and create cross-domain concept nodes.

    Rules with provenance domains (e.g., "observer_pattern", "implementation_hiding")
    become Concept nodes in their source domain, bridging experiential data to
    design pattern knowledge.

    Returns:
        Tuple of (concept_nodes, edges) ready for graph ingestion.
    """
    import sqlite3

    if not db_path.exists():
        return [], []

    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row

    concepts: list[ConceptNode] = []
    edges: list[ConceptEdge] = []
    seen_ids: set[str] = set()

    cursor = db.execute(
        "SELECT rule_id, rule, category, provenance, seed_filename, active FROM gauntlet_rules"
    )
    for row in cursor:
        rule_id = row["rule_id"]
        rule_text = row["rule"]
        category = row["category"] or "general"
        active = bool(row["active"])
        seed_file = row["seed_filename"] or ""

        # Parse provenance for domain
        prov = {}
        if row["provenance"]:
            try:
                prov = json.loads(row["provenance"])
            except json.JSONDecodeError:
                pass

        source_domain = prov.get("domain", "")
        derivation = prov.get("derivation", "")

        # Skip inactive rules and edge-template derived junk
        if not active:
            continue
        if derivation == "derived":
            continue

        # Skip rules without a known design-pattern domain
        if not source_domain or source_domain in ("", "unknown", "experiential", "buildlog"):
            # Still create a concept for non-domain rules (personas like test_terrorist)
            persona = rule_id.split(":")[0] if ":" in rule_id else seed_file.replace(".yaml", "")
            if persona and persona not in seen_ids:
                seen_ids.add(persona)
                concepts.append(ConceptNode(
                    id=f"persona:{persona}",
                    name=persona,
                    description=f"Gauntlet reviewer persona from {seed_file}",
                    domain="buildlog",
                    source_id="buildlog:gauntlet",
                    properties={"type": "persona", "seed_file": seed_file},
                ))

            # Create rule concept
            if rule_id not in seen_ids:
                seen_ids.add(rule_id)
                concepts.append(ConceptNode(
                    id=f"gauntlet_rule:{rule_id}",
                    name=rule_text[:80],
                    description=f"[{category}] {rule_text}",
                    domain="buildlog",
                    source_id="buildlog:gauntlet",
                    properties={
                        "category": category,
                        "active": active,
                        "persona": persona,
                    },
                ))
                # Edge: rule belongs_to persona
                edges.append(ConceptEdge(
                    source_id=f"gauntlet_rule:{rule_id}",
                    target_id=f"persona:{persona}",
                    relation_type=RelationType.BELONGS_TO,
                    confidence=1.0,
                ))
            continue

        # This rule has a design pattern domain — it's a BRIDGE
        concept_id = f"gauntlet_rule:{rule_id}"
        if concept_id in seen_ids:
            continue
        seen_ids.add(concept_id)

        # Create concept in the SOURCE domain (design patterns)
        concepts.append(ConceptNode(
            id=concept_id,
            name=rule_text[:80],
            description=f"[{category}] {rule_text}",
            domain=source_domain,  # KEY: this goes in the design pattern domain
            source_id="buildlog:gauntlet",
            properties={
                "category": category,
                "active": active,
                "source_domain": source_domain,
                "derivation": prov.get("derivation", "unknown"),
                "confidence": prov.get("confidence", 0.5),
                "seed_file": seed_file,
                "bridge": True,  # marks this as a cross-domain bridge node
            },
        ))

        # Edge: rule instance_of its domain (cross-domain bridge)
        domain_concept_id = f"domain:{source_domain}"
        if domain_concept_id not in seen_ids:
            seen_ids.add(domain_concept_id)
            concepts.append(ConceptNode(
                id=domain_concept_id,
                name=source_domain.replace("_", " ").title(),
                description=f"Design pattern domain: {source_domain}",
                domain=source_domain,
                source_id="buildlog:gauntlet",
                properties={"type": "domain_anchor"},
            ))

        edges.append(ConceptEdge(
            source_id=concept_id,
            target_id=domain_concept_id,
            relation_type=RelationType.INSTANCE_OF,
            confidence=1.0,
        ))

    db.close()
    return concepts, edges
