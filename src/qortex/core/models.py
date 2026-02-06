"""Core data models for the knowledge graph.

These models define the contract between components:
- Ingestors produce IngestionManifest
- KG consumes manifests and stores Nodes/Edges
- Projectors read from KG and produce Rules
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal


# =============================================================================
# Ingestion Manifest (Contract: Ingestors → KG)
# =============================================================================
# This is the BOUNDARY between ingest and the KG.
# Ingestors (PDF, MD, Text) produce manifests.
# The KG consumes manifests - it doesn't know about source formats.
# This allows ingest to be a separate package/service later.


class RelationType(str, Enum):
    """Semantic relationship types between concepts.

    Rich typing from the start enables Phase B rule derivation.
    """
    # Logical relationships
    CONTRADICTS = "contradicts"      # A and B are mutually exclusive
    REQUIRES = "requires"            # A requires B to be true/present
    REFINES = "refines"              # A is a more specific form of B
    IMPLEMENTS = "implements"        # A is a concrete implementation of B

    # Compositional relationships
    PART_OF = "part_of"              # A is a component of B
    USES = "uses"                    # A uses/depends on B

    # Similarity relationships
    SIMILAR_TO = "similar_to"        # A and B are related/analogous
    ALTERNATIVE_TO = "alternative_to"  # A can substitute for B

    # Epistemic relationships
    SUPPORTS = "supports"            # A provides evidence for B
    CHALLENGES = "challenges"        # A provides counter-evidence for B


@dataclass
class ConceptNode:
    """A concept extracted from source material."""
    id: str
    name: str
    description: str
    domain: str  # Which domain this belongs to

    # Source provenance
    source_id: str
    source_location: str | None = None  # e.g., "chapter 3, page 45"

    # Metadata
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0  # Extraction confidence


@dataclass
class ConceptEdge:
    """A relationship between concepts."""
    source_id: str
    target_id: str
    relation_type: RelationType

    # Edge metadata
    properties: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    bidirectional: bool = False


@dataclass
class CodeExample:
    """A code example linked to concepts and rules.

    Structure matches future SQLA model for direct deserialization:
        CodeExample(**example_dict) works for both dataclass and SQLA.

    Used for:
    - Few-shot prompting: Rule -> Examples -> prompt context
    - 2nd-order retrieval: Query -> similar code -> linked concepts
    - Contrastive learning: Good example vs antipattern
    """
    id: str
    code: str
    language: str
    description: str | None = None
    source_location: str | None = None  # e.g., "ch11:p42"

    # Links (M2M in SQLA, stored as IDs here)
    concept_ids: list[str] = field(default_factory=list)
    rule_ids: list[str] = field(default_factory=list)

    # Classification
    tags: list[str] = field(default_factory=list)
    is_antipattern: bool = False  # For contrastive learning

    # Extensible
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExplicitRule:
    """A rule explicitly stated in source material.

    Phase C: Rules are explicit, linked to concepts.
    Phase B (future): Rules derived from edges via templates.
    """
    id: str
    text: str
    domain: str
    source_id: str

    # Links to concepts this rule operationalizes
    concept_ids: list[str] = field(default_factory=list)

    # Source provenance
    source_location: str | None = None

    # Metadata
    category: str | None = None  # e.g., "architectural", "testing"
    confidence: float = 1.0


@dataclass
class SourceMetadata:
    """Metadata about an ingested source."""
    id: str
    name: str
    source_type: Literal["pdf", "markdown", "text", "url"]
    path_or_url: str

    # Ingestion info
    ingested_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    content_hash: str | None = None  # For change detection

    # Statistics
    chunk_count: int = 0
    concept_count: int = 0
    rule_count: int = 0


@dataclass
class IngestionManifest:
    """The output of an ingestor - consumed by the KG.

    This is the CONTRACT between ingest and storage.
    The KG doesn't know about PDFs or Markdown - it just knows manifests.
    """
    source: SourceMetadata
    domain: str

    # Extracted content
    concepts: list[ConceptNode]
    edges: list[ConceptEdge]
    rules: list[ExplicitRule]
    examples: list[CodeExample] = field(default_factory=list)

    # Quality metrics
    extraction_confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)


# =============================================================================
# Domain Model
# =============================================================================


@dataclass
class Domain:
    """A domain is an isolated subgraph - like a Postgres schema."""
    name: str
    description: str | None = None

    # Sources that contributed to this domain
    source_ids: list[str] = field(default_factory=list)

    # Stats
    concept_count: int = 0
    edge_count: int = 0
    rule_count: int = 0

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Rule (as projected for consumers)
# =============================================================================


@dataclass
class Rule:
    """A rule as returned to consumers (buildlog, agents, etc.)."""
    id: str
    text: str
    domain: str

    # How this rule was produced
    derivation: Literal["explicit", "derived"]  # Phase C vs Phase B
    source_concepts: list[str]  # Concept IDs it came from

    # Scoring
    confidence: float
    relevance: float = 0.0  # Set by retrieval (e.g., PPR score)

    # Category for filtering
    category: str | None = None

    # Extensible metadata (e.g. template info for derived rules)
    metadata: dict[str, Any] = field(default_factory=dict)


