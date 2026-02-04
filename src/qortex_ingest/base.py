"""Base ingestor protocol and types.

Ingestors transform raw sources into IngestionManifest.
The manifest is the CONTRACT - KG doesn't know about source formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from qortex.core.models import (
    ConceptEdge,
    ConceptNode,
    ExplicitRule,
    IngestionManifest,
    SourceMetadata,
)

if TYPE_CHECKING:
    from .llm import LLMBackend


@dataclass
class Chunk:
    """A chunk of source content for processing."""
    id: str
    content: str

    # Location in source
    location: str | None = None  # e.g., "Chapter 3, Section 2"
    page: int | None = None

    # Hierarchy
    parent_id: str | None = None  # For nested chunks (chapter > section)
    level: int = 0  # 0 = top level


@dataclass
class Source:
    """Input source to be ingested."""
    path: Path | None = None
    url: str | None = None
    raw_content: str | None = None  # For paste-in content

    source_type: Literal["pdf", "markdown", "text", "url"] = "text"
    name: str | None = None  # Human-readable name

    def __post_init__(self):
        if not any([self.path, self.url, self.raw_content]):
            raise ValueError("Source must have path, url, or raw_content")


class Ingestor(ABC):
    """Base class for source ingestors.

    Subclasses implement format-specific chunking.
    LLM extraction is shared across all ingestors.
    """

    def __init__(self, llm: "LLMBackend"):
        self.llm = llm

    @abstractmethod
    def chunk(self, source: Source) -> list[Chunk]:
        """Split source into chunks for processing.

        Format-specific: PDF chunks by page/section, MD by heading, etc.
        """
        ...

    def ingest(
        self,
        source: Source,
        domain: str | None = None,
    ) -> IngestionManifest:
        """Full ingestion pipeline: chunk → extract → manifest.

        If domain is None, LLM suggests a name.
        """
        # 1. Chunk the source
        chunks = self.chunk(source)

        # 2. Determine domain
        if domain is None:
            sample = chunks[0].content[:1000] if chunks else ""
            domain = self.llm.suggest_domain_name(
                source.name or "unknown",
                sample,
            )

        # 3. Extract concepts from chunks
        concepts: list[ConceptNode] = []
        source_id = f"{domain}:{source.name or 'source'}"

        for chunk in chunks:
            extracted = self.llm.extract_concepts(chunk.content, domain)
            for c in extracted:
                concepts.append(ConceptNode(
                    id=f"{domain}:{c['name']}",
                    name=c["name"],
                    description=c.get("description", ""),
                    domain=domain,
                    source_id=source_id,
                    source_location=chunk.location,
                    confidence=c.get("confidence", 1.0),
                ))

        # 4. Extract relations between concepts
        all_text = "\n\n".join(c.content for c in chunks)
        relation_dicts = self.llm.extract_relations(concepts, all_text)
        edges = [
            ConceptEdge(
                source_id=r["source_id"],
                target_id=r["target_id"],
                relation_type=r["relation_type"],
                confidence=r.get("confidence", 1.0),
            )
            for r in relation_dicts
        ]

        # 5. Extract explicit rules
        rule_dicts = self.llm.extract_rules(all_text, concepts)
        rules = [
            ExplicitRule(
                id=f"{domain}:rule:{i}",
                text=r["text"],
                domain=domain,
                concept_ids=r.get("concept_ids", []),
                source_id=source_id,
                category=r.get("category"),
                confidence=r.get("confidence", 1.0),
            )
            for i, r in enumerate(rule_dicts)
        ]

        # 6. Build manifest
        source_meta = SourceMetadata(
            id=source_id,
            name=source.name or "unknown",
            source_type=source.source_type,
            path_or_url=str(source.path or source.url or "raw"),
            chunk_count=len(chunks),
            concept_count=len(concepts),
            rule_count=len(rules),
        )

        return IngestionManifest(
            source=source_meta,
            domain=domain,
            concepts=concepts,
            edges=edges,
            rules=rules,
        )


