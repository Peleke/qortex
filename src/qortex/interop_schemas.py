"""JSON Schema definitions for the consumer interop protocol.

These schemas are the canonical contract. Any consumer in any language
can validate against them without importing Python code.

Schemas are versioned independently of the qortex package version.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Schema versions
SEED_SCHEMA_VERSION = "1.0"
EVENT_SCHEMA_VERSION = "1.0"

# =============================================================================
# Seed Schema (Universal Rule Set Format)
# =============================================================================

SEED_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://qortex.dev/schemas/seed.v1.schema.json",
    "title": "Qortex Seed",
    "description": "Universal rule set format for agent consumption. Any consumer that understands this schema can ingest qortex projections.",
    "type": "object",
    "required": ["persona", "version", "rules", "metadata"],
    "properties": {
        "persona": {
            "type": "string",
            "description": "Flat string identifier. Consumers may use as filename.",
            "minLength": 1,
            "examples": ["qortex_implementation_hiding", "security_rules"],
        },
        "version": {
            "type": "integer",
            "description": "Schema version as integer. Consumers compare numerically.",
            "minimum": 1,
            "examples": [1, 2],
        },
        "rules": {
            "type": "array",
            "description": "List of rules with enrichment and provenance.",
            "items": {"$ref": "#/$defs/rule"},
        },
        "metadata": {"$ref": "#/$defs/metadata"},
    },
    "$defs": {
        "rule": {
            "type": "object",
            "required": ["rule", "category", "provenance"],
            "properties": {
                "rule": {"type": "string", "description": "The rule text itself.", "minLength": 1},
                "category": {
                    "type": "string",
                    "description": "Rule category for filtering. Falls back to domain if not specified.",
                    "examples": ["architectural", "testing", "security"],
                },
                "context": {
                    "type": "string",
                    "description": "When this rule applies (from enrichment).",
                },
                "antipattern": {
                    "type": "string",
                    "description": "What violating this rule looks like (from enrichment).",
                },
                "rationale": {
                    "type": "string",
                    "description": "Why this rule matters (from enrichment).",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Searchable tags (from enrichment).",
                },
                "provenance": {"$ref": "#/$defs/provenance"},
            },
        },
        "provenance": {
            "type": "object",
            "description": "Origin and derivation metadata. Opaque to most consumers.",
            "required": ["id", "domain", "derivation", "confidence"],
            "properties": {
                "id": {"type": "string", "description": "Unique rule identifier."},
                "domain": {
                    "type": "string",
                    "description": "Knowledge domain this rule belongs to.",
                },
                "derivation": {
                    "type": "string",
                    "enum": ["explicit", "derived"],
                    "description": "Whether rule was explicit in source or derived from edges.",
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score (0-1). Consumers may use for ranking.",
                },
                "relevance": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Relevance score from retrieval (e.g., PPR).",
                },
                "source_concepts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concept IDs this rule derives from.",
                },
                "relation_type": {
                    "type": ["string", "null"],
                    "description": "Edge relation type (for derived rules).",
                    "examples": ["requires", "contradicts", "supports"],
                },
                "template_id": {
                    "type": ["string", "null"],
                    "description": "Template ID used for derivation.",
                },
                "template_variant": {
                    "type": ["string", "null"],
                    "description": "Template variant (imperative, conditional, warning).",
                },
                "template_severity": {
                    "type": ["string", "null"],
                    "description": "Template severity level.",
                },
                "graph_version": {
                    "type": ["string", "null"],
                    "description": "ISO timestamp of graph state used for projection.",
                },
            },
        },
        "metadata": {
            "type": "object",
            "required": ["source", "rule_count"],
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Origin system identifier.",
                    "examples": ["qortex", "manual"],
                },
                "source_version": {
                    "type": "string",
                    "description": "Version of the source system.",
                },
                "projected_at": {
                    "type": "string",
                    "format": "date-time",
                    "description": "ISO timestamp of projection.",
                },
                "rule_count": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Number of rules in this seed.",
                },
            },
        },
    },
}

# =============================================================================
# Signal Event Schema
# =============================================================================

EVENT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://qortex.dev/schemas/event.v1.schema.json",
    "title": "Qortex Interop Event",
    "description": "Signal log event for the consumer interop protocol. Append-only JSONL format.",
    "type": "object",
    "required": ["event", "ts", "source"],
    "properties": {
        "event": {
            "type": "string",
            "description": "Event type identifier.",
            "enum": ["projection_complete", "seed_ingested", "seed_failed"],
            "examples": ["projection_complete"],
        },
        "persona": {"type": "string", "description": "Persona identifier for the seed."},
        "domain": {"type": "string", "description": "Domain that was projected."},
        "path": {"type": "string", "description": "Filesystem path to the seed file."},
        "rule_count": {
            "type": "integer",
            "minimum": 0,
            "description": "Number of rules in the projection.",
        },
        "ts": {
            "type": "string",
            "format": "date-time",
            "description": "ISO timestamp of the event.",
        },
        "source": {
            "type": "string",
            "description": "System that emitted the event.",
            "examples": ["qortex", "buildlog"],
        },
        "source_version": {"type": "string", "description": "Version of the emitting system."},
        "error": {"type": "string", "description": "Error message (for seed_failed events)."},
    },
    "additionalProperties": True,
}


# =============================================================================
# Export Functions
# =============================================================================


def get_seed_schema() -> dict[str, Any]:
    """Get the seed schema as a dict."""
    return SEED_SCHEMA.copy()


def get_event_schema() -> dict[str, Any]:
    """Get the event schema as a dict."""
    return EVENT_SCHEMA.copy()


def export_schemas(directory: Path | str) -> tuple[Path, Path]:
    """Export schemas to JSON files in the given directory.

    Args:
        directory: Directory to write schema files to.

    Returns:
        Tuple of (seed_schema_path, event_schema_path).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    seed_path = directory / f"seed.v{SEED_SCHEMA_VERSION}.schema.json"
    event_path = directory / f"event.v{EVENT_SCHEMA_VERSION}.schema.json"

    seed_path.write_text(json.dumps(SEED_SCHEMA, indent=2))
    event_path.write_text(json.dumps(EVENT_SCHEMA, indent=2))

    return seed_path, event_path


def validate_seed(seed: dict[str, Any]) -> list[str]:
    """Validate a seed against the schema.

    Returns list of validation errors (empty if valid).
    Requires jsonschema package for full validation.
    Falls back to basic structural checks if not available.
    """
    errors = []

    # Basic structural validation (always available)
    if not isinstance(seed, dict):
        return ["Seed must be a dict"]

    for field in ["persona", "version", "rules", "metadata"]:
        if field not in seed:
            errors.append(f"Missing required field: {field}")

    if "persona" in seed and not isinstance(seed["persona"], str):
        errors.append("persona must be a string")

    if "version" in seed and not isinstance(seed["version"], int):
        errors.append("version must be an integer")

    if "rules" in seed and not isinstance(seed["rules"], list):
        errors.append("rules must be a list")

    # Try full JSON Schema validation if available
    try:
        import jsonschema

        validator = jsonschema.Draft202012Validator(SEED_SCHEMA)
        for error in validator.iter_errors(seed):
            errors.append(f"{error.json_path}: {error.message}")
    except ImportError:
        pass  # Basic validation only

    return errors


def validate_event(event: dict[str, Any]) -> list[str]:
    """Validate an event against the schema.

    Returns list of validation errors (empty if valid).
    """
    errors = []

    if not isinstance(event, dict):
        return ["Event must be a dict"]

    for field in ["event", "ts", "source"]:
        if field not in event:
            errors.append(f"Missing required field: {field}")

    if "event" in event:
        valid_events = ["projection_complete", "seed_ingested", "seed_failed"]
        if event["event"] not in valid_events:
            errors.append(f"Invalid event type: {event['event']}")

    # Try full JSON Schema validation if available
    try:
        import jsonschema

        validator = jsonschema.Draft202012Validator(EVENT_SCHEMA)
        for error in validator.iter_errors(event):
            errors.append(f"{error.json_path}: {error.message}")
    except ImportError:
        pass

    return errors
