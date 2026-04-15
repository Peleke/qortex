"""SkillMdIngestor -- deterministic SKILL.md to IngestionManifest.

No LLM required. Parses SKILL.md files using the skillmd parser and produces
IngestionManifest objects for direct consumption by the knowledge graph.

Supports single-file and batch directory ingestion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from qortex.core.models import (
    ConceptEdge,
    ConceptNode,
    ExplicitRule,
    IngestionManifest,
    RelationType,
    SourceMetadata,
)
from qortex.projectors.skillmd import (
    SkillMdDocument,
    _extract_sections,
    parse_skill_md,
)


@dataclass
class SkillMdIngestor:
    """Deterministic SKILL.md ingestor -- no LLM required.

    Parses SKILL.md files and produces IngestionManifest objects containing
    concept nodes, edges, and explicit rules extracted from skill structure.
    """

    domain_prefix: str = "skill"
    default_confidence: float = 0.75

    def ingest(
        self, path: Path, domain: str | None = None
    ) -> IngestionManifest:
        """Parse a single SKILL.md file into an IngestionManifest.

        Args:
            path: Path to the SKILL.md file.
            domain: Optional domain override. Defaults to ``{domain_prefix}:{name}``.

        Returns:
            IngestionManifest ready for KG consumption.

        Raises:
            ValueError: If the file cannot be parsed.
        """
        content = path.read_text(encoding="utf-8")
        doc = parse_skill_md(content, source_path=path)
        resolved_domain = domain or f"{self.domain_prefix}:{doc.name}"
        return self._build_manifest(doc, resolved_domain)

    def ingest_directory(
        self,
        directory: Path,
        domain_prefix: str | None = None,
        recursive: bool = True,
    ) -> list[IngestionManifest]:
        """Batch-ingest all SKILL.md files in a directory.

        Args:
            directory: Directory to scan.
            domain_prefix: Override for the domain prefix. Defaults to
                ``self.domain_prefix``.
            recursive: If True, search subdirectories.

        Returns:
            List of IngestionManifest objects, one per SKILL.md file found.
        """
        prefix = domain_prefix or self.domain_prefix
        pattern = "**/SKILL.md" if recursive else "SKILL.md"
        manifests: list[IngestionManifest] = []

        for skill_path in sorted(directory.glob(pattern)):
            content = skill_path.read_text(encoding="utf-8")
            doc = parse_skill_md(content, source_path=skill_path)
            domain = f"{prefix}:{doc.name}"
            manifests.append(self._build_manifest(doc, domain))

        return manifests

    def _build_manifest(
        self, doc: SkillMdDocument, domain: str
    ) -> IngestionManifest:
        """Core logic: convert a parsed SkillMdDocument into an IngestionManifest.

        Manifest structure:
        1. Root ConceptNode for the skill itself.
        2. Section ConceptNodes for each ## heading in the body.
        3. PART_OF edges from sections to root.
        4. Primary ExplicitRule from skill description.
        5. Per-section ExplicitRules for instructional sections.
        6. SourceMetadata with source_type="skill_md".
        """
        root_id = f"skill:{doc.name}"
        source_id = doc.skill_id

        # -- Root concept node --
        properties: dict = {
            "skill_format": doc.source_format,
            "content_hash": doc.content_hash,
            "body": doc.body,
        }
        if doc.homepage:
            properties["homepage"] = doc.homepage
        if doc.openclaw_metadata:
            properties["openclaw"] = doc.openclaw_metadata

        root_node = ConceptNode(
            id=root_id,
            name=doc.name,
            description=doc.description,
            domain=domain,
            source_id=source_id,
            source_location=str(doc.source_path) if doc.source_path else None,
            properties=properties,
            confidence=self.default_confidence,
        )

        concepts: list[ConceptNode] = [root_node]
        edges: list[ConceptEdge] = []
        rules: list[ExplicitRule] = []

        # -- Section concept nodes + PART_OF edges --
        sections = _extract_sections(doc.body)
        for i, section in enumerate(sections):
            section_id = f"skill:{doc.name}:section:{i}"
            section_content = section["content"]
            description = section_content[:200] if section_content else ""

            section_node = ConceptNode(
                id=section_id,
                name=section["heading"] or f"preamble",
                description=description,
                domain=domain,
                source_id=source_id,
                source_location=section["heading"] or "preamble",
                confidence=self.default_confidence,
            )
            concepts.append(section_node)

            edges.append(
                ConceptEdge(
                    source_id=section_id,
                    target_id=root_id,
                    relation_type=RelationType.PART_OF,
                    confidence=self.default_confidence,
                )
            )

        # -- Primary explicit rule from skill description --
        primary_rule = ExplicitRule(
            id=f"skill_rule:{doc.name}:instructions",
            text=f"Skill '{doc.name}': {doc.description}",
            domain=domain,
            source_id=source_id,
            concept_ids=[root_id],
            confidence=self.default_confidence,
        )
        rules.append(primary_rule)

        # -- Per-section rules for instructional sections --
        for i, section in enumerate(sections):
            if self._is_instructional(section["content"]):
                heading = section["heading"] or "preamble"
                section_id = f"skill:{doc.name}:section:{i}"
                section_rule = ExplicitRule(
                    id=f"skill_rule:{doc.name}:section:{i}",
                    text=section["content"],
                    domain=domain,
                    source_id=source_id,
                    concept_ids=[section_id, root_id],
                    confidence=self.default_confidence,
                    category=heading,
                )
                rules.append(section_rule)

        # -- Source metadata --
        source_meta = SourceMetadata(
            id=source_id,
            name=doc.name,
            source_type="skill_md",
            path_or_url=str(doc.source_path) if doc.source_path else "",
            content_hash=doc.content_hash,
            concept_count=len(concepts),
            rule_count=len(rules),
        )

        return IngestionManifest(
            source=source_meta,
            domain=domain,
            concepts=concepts,
            edges=edges,
            rules=rules,
            extraction_confidence=self.default_confidence,
        )

    @staticmethod
    def _is_instructional(content: str) -> bool:
        """Check if section content is instructional.

        A section is considered instructional if it contains code blocks,
        markdown lists, or imperative verbs.
        """
        if not content or not content.strip():
            return False

        # Code blocks (fenced or indented)
        if re.search(r"```", content):
            return True

        # Markdown lists (- item, * item, 1. item)
        if re.search(r"^\s*[-*]\s+\S", content, re.MULTILINE):
            return True
        if re.search(r"^\s*\d+\.\s+\S", content, re.MULTILINE):
            return True

        # Imperative verbs at start of sentences
        imperative_pattern = (
            r"(?:^|\.\s+)"
            r"(?:Use|Avoid|Always|Never|Ensure|Make sure|Do not|Don't|"
            r"Consider|Prefer|Keep|Set|Run|Add|Remove|Check|Create|"
            r"Include|Exclude|Follow|Apply|Configure|Define|Implement|"
            r"Return|Pass|Call|Handle|Wrap|Extract|Import|Export)"
            r"\s"
        )
        if re.search(imperative_pattern, content, re.MULTILINE | re.IGNORECASE):
            return True

        return False
