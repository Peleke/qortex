"""Base ingestor protocol and types.

Ingestors transform raw sources into IngestionManifest.
The manifest is the CONTRACT - KG doesn't know about source formats.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qortex.core.pruning import PruningConfig

from qortex.core.models import (
    CodeExample,
    ConceptEdge,
    ConceptNode,
    ExplicitRule,
    IngestionManifest,
    SourceMetadata,
)


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


@runtime_checkable
class LLMBackend(Protocol):
    """LLM backend for extraction tasks."""

    def extract_concepts(self, text: str, domain_hint: str | None = None) -> list[dict]:
        """Extract concepts from text."""
        ...

    def extract_relations(
        self,
        concepts: list[ConceptNode],
        text: str,
        chunk_location: str | None = None,
    ) -> list[dict]:
        """Extract relations between concepts."""
        ...

    def extract_rules(self, text: str, concepts: list[ConceptNode]) -> list[dict]:
        """Extract explicit rules from text."""
        ...

    def extract_code_examples(
        self,
        text: str,
        concepts: list[ConceptNode],
        domain: str,
    ) -> list[dict]:
        """Extract code examples and link to concepts.

        Optional method - returns empty list if not implemented.
        """
        ...

    def suggest_domain_name(self, source_name: str, sample_text: str) -> str:
        """Suggest a domain name for a source."""
        ...


class Ingestor(ABC):
    """Base class for source ingestors.

    Subclasses implement format-specific chunking.
    LLM extraction is shared across all ingestors.
    """

    def __init__(
        self,
        llm: LLMBackend,
        pruning_config: PruningConfig | None = None,
    ):
        from qortex.core.pruning import PruningConfig

        self.llm = llm
        self.pruning_config = pruning_config or PruningConfig()

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

        # 3. Extract concepts from chunks (two-pass: generalizable first, then reconcile illustrative)
        source_id = f"{domain}:{source.name or 'source'}"

        # Pass 1: Collect all extracted concepts, separating by role
        generalizable_concepts: list[ConceptNode] = []
        illustrative_raw: list[tuple[dict, str | None]] = []  # (raw_dict, chunk_location)

        for chunk in chunks:
            extracted = self.llm.extract_concepts(chunk.content, domain)
            for c in extracted:
                if c.get("concept_role") == "illustrative":
                    illustrative_raw.append((c, chunk.location))
                else:
                    generalizable_concepts.append(
                        ConceptNode(
                            id=f"{domain}:{c['name']}",
                            name=c["name"],
                            description=c.get("description", ""),
                            domain=domain,
                            source_id=source_id,
                            source_location=chunk.location,
                            confidence=c.get("confidence", 1.0),
                        )
                    )

        # Pass 2: Reconcile illustrative concepts → parent properties["examples"]
        concept_by_name: dict[str, ConceptNode] = {
            c.name.lower(): c for c in generalizable_concepts
        }

        for raw, location in illustrative_raw:
            parent_name = raw.get("illustrates")
            parent = concept_by_name.get(parent_name.lower()) if parent_name else None

            if parent is not None:
                # Attach as example on the parent concept
                if "examples" not in parent.properties:
                    parent.properties["examples"] = []
                parent.properties["examples"].append(
                    {
                        "name": raw["name"],
                        "description": raw.get("description", ""),
                        "source_location": location,
                        "confidence": raw.get("confidence", 1.0),
                    }
                )
            else:
                # Parent not found — create as concept but mark role in properties
                generalizable_concepts.append(
                    ConceptNode(
                        id=f"{domain}:{raw['name']}",
                        name=raw["name"],
                        description=raw.get("description", ""),
                        domain=domain,
                        source_id=source_id,
                        source_location=location,
                        confidence=raw.get("confidence", 1.0),
                        properties={
                            "concept_role": "illustrative",
                            "illustrates": parent_name,
                        },
                    )
                )

        concepts = generalizable_concepts

        # 4. Extract relations per chunk (for full coverage + provenance)
        from qortex.core.models import RelationType
        from qortex.core.pruning import PruningConfig, prune_edges

        all_relations: list[dict] = []
        seen_edges: set[tuple[str, str, str]] = set()  # Dedupe across chunks

        for chunk in chunks:
            relation_dicts = self.llm.extract_relations(
                concepts, chunk.content, chunk_location=chunk.location
            )
            for r in relation_dicts:
                # Dedupe: same source->target->type only counted once
                rel_type_str = (
                    r["relation_type"].lower()
                    if isinstance(r["relation_type"], str)
                    else r["relation_type"]
                )
                edge_key = (r["source_id"], r["target_id"], rel_type_str)
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)
                all_relations.append(r)

        # 4b. Prune edges (online mode, enabled by default)
        pruning_config = getattr(self, "pruning_config", None) or PruningConfig()
        prune_result = prune_edges(all_relations, pruning_config)
        pruned_relations = prune_result.edges

        # Convert to ConceptEdge objects
        edges = []
        for r in pruned_relations:
            rel_type = r["relation_type"]
            if isinstance(rel_type, str):
                try:
                    rel_type = RelationType(rel_type.lower())
                except ValueError:
                    continue  # Skip invalid relation types

            # Store provenance + pruning metadata in properties
            properties = {}
            if r.get("source_text"):
                properties["source_text"] = r["source_text"]
            if r.get("source_location"):
                properties["source_location"] = r["source_location"]
            if r.get("layer"):
                properties["layer"] = r["layer"]
            if r.get("strength"):
                properties["strength"] = r["strength"]

            edges.append(
                ConceptEdge(
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    relation_type=rel_type,
                    confidence=r.get("confidence", 1.0),
                    properties=properties,
                )
            )

        # 5. Extract explicit rules (use first few chunks for rule extraction)
        all_text = "\n\n".join(chunk.content for chunk in chunks[:5])
        rule_dicts = self.llm.extract_rules(all_text, concepts[:50])
        valid_concept_ids = {c.id for c in concepts}
        rules = [
            ExplicitRule(
                id=f"{domain}:rule:{i}",
                text=r["text"],
                domain=domain,
                concept_ids=[cid for cid in r.get("concept_ids", []) if cid in valid_concept_ids],
                source_id=source_id,
                category=r.get("category"),
                confidence=r.get("confidence", 1.0),
            )
            for i, r in enumerate(rule_dicts)
        ]

        # 6. Extract code examples (if backend supports it)
        examples: list[CodeExample] = []
        if hasattr(self.llm, "extract_code_examples"):
            example_dicts = self.llm.extract_code_examples(all_text, concepts[:50], domain)
            for ex in example_dicts:
                examples.append(
                    CodeExample(
                        id=ex["id"],
                        code=ex["code"],
                        language=ex["language"],
                        description=ex.get("description"),
                        source_location=ex.get("source_location"),
                        concept_ids=ex.get("concept_ids", []),
                        rule_ids=ex.get("rule_ids", []),
                        tags=ex.get("tags", []),
                        is_antipattern=ex.get("is_antipattern", False),
                        properties=ex.get("properties", {}),
                    )
                )

        # 7. Build manifest
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
            examples=examples,
        )


# =============================================================================
# Stub LLM Backend (for testing without real LLM)
# =============================================================================


class StubLLMBackend:
    """Stub LLM that returns configurable results.

    Use for testing pipeline without LLM costs.
    Pass concepts/relations/rules to constructor to inject test data.
    """

    def __init__(
        self,
        concepts: list[dict] | None = None,
        relations: list[dict] | None = None,
        rules: list[dict] | None = None,
    ):
        self._concepts = concepts or []
        self._relations = relations or []
        self._rules = rules or []

    def extract_concepts(self, text: str, domain_hint: str | None = None) -> list[dict]:
        return self._concepts

    def extract_relations(
        self,
        concepts: list[ConceptNode],
        text: str,
        chunk_location: str | None = None,
    ) -> list[dict]:
        return self._relations

    def extract_rules(self, text: str, concepts: list[ConceptNode]) -> list[dict]:
        return self._rules

    def extract_code_examples(
        self,
        text: str,
        concepts: list[ConceptNode],
        domain: str,
    ) -> list[dict]:
        return []

    def suggest_domain_name(self, source_name: str, sample_text: str) -> str:
        return source_name.lower().replace(" ", "_")
