"""Output strategies for different targets.

Each strategy knows how to:
1. Accept extraction results (concepts, edges, rules)
2. Transform/write to target format
3. Return appropriate result type
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Type, TypeVar

import yaml

from qortex.core.models import (
    ConceptEdge,
    ConceptNode,
    ExplicitRule,
    IngestionManifest,
    SourceMetadata,
)

T = TypeVar("T")


@dataclass
class ExtractionResult:
    """Intermediate extraction results before output formatting."""
    source: SourceMetadata
    domain: str
    concepts: list[ConceptNode]
    edges: list[ConceptEdge]
    rules: list[ExplicitRule]


class OutputStrategy(ABC, Generic[T]):
    """Base class for output format strategies.

    Subclasses implement format-specific transformation and writing.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g., 'manifest', 'json', 'yaml')."""
        ...

    @abstractmethod
    def output(self, result: ExtractionResult) -> T:
        """Transform extraction result to target format."""
        ...


# =============================================================================
# Registry
# =============================================================================

_OUTPUT_STRATEGIES: dict[str, Type[OutputStrategy]] = {}


def register_output_strategy(name: str, strategy_class: Type[OutputStrategy]) -> None:
    """Register an output strategy."""
    _OUTPUT_STRATEGIES[name] = strategy_class


def available_output_strategies() -> list[str]:
    """List registered output strategies."""
    _ensure_registered()
    return list(_OUTPUT_STRATEGIES.keys())


def get_output_strategy(name: str = "manifest", **kwargs) -> OutputStrategy:
    """Get an output strategy.

    Args:
        name: Strategy name (e.g., 'manifest', 'json', 'yaml', 'kg')
        **kwargs: Strategy-specific configuration

    Returns:
        Configured OutputStrategy instance
    """
    _ensure_registered()

    if name not in _OUTPUT_STRATEGIES:
        raise ValueError(f"Unknown output strategy: {name}. Available: {list(_OUTPUT_STRATEGIES.keys())}")

    return _OUTPUT_STRATEGIES[name](**kwargs)


# =============================================================================
# Built-in Strategies
# =============================================================================


class ManifestOutputStrategy(OutputStrategy[IngestionManifest]):
    """Output as IngestionManifest (default)."""

    @property
    def name(self) -> str:
        return "manifest"

    def output(self, result: ExtractionResult) -> IngestionManifest:
        """Return IngestionManifest directly."""
        return IngestionManifest(
            source=result.source,
            domain=result.domain,
            concepts=result.concepts,
            edges=result.edges,
            rules=result.rules,
        )


class JSONOutputStrategy(OutputStrategy[str]):
    """Output as JSON string."""

    def __init__(self, indent: int = 2, include_metadata: bool = True):
        self.indent = indent
        self.include_metadata = include_metadata

    @property
    def name(self) -> str:
        return "json"

    def output(self, result: ExtractionResult) -> str:
        """Return JSON string."""
        data = {
            "domain": result.domain,
            "concepts": [
                {
                    "id": c.id,
                    "name": c.name,
                    "description": c.description,
                    "confidence": c.confidence,
                }
                for c in result.concepts
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "relation": e.relation_type.value,
                    "confidence": e.confidence,
                }
                for e in result.edges
            ],
            "rules": [
                {
                    "id": r.id,
                    "text": r.text,
                    "category": r.category,
                    "confidence": r.confidence,
                }
                for r in result.rules
            ],
        }

        if self.include_metadata:
            data["metadata"] = {
                "source_id": result.source.id,
                "source_name": result.source.name,
                "source_type": result.source.source_type,
                "extracted_at": datetime.utcnow().isoformat(),
            }

        return json.dumps(data, indent=self.indent)


class YAMLOutputStrategy(OutputStrategy[str]):
    """Output as YAML string (buildlog-compatible format)."""

    def __init__(self, rules_only: bool = False):
        self.rules_only = rules_only

    @property
    def name(self) -> str:
        return "yaml"

    def output(self, result: ExtractionResult) -> str:
        """Return YAML string."""
        if self.rules_only:
            # Buildlog-compatible format
            data = {
                "domain": result.domain,
                "rules": [
                    {
                        "id": r.id,
                        "text": r.text,
                        "category": r.category or "general",
                        "confidence": r.confidence,
                    }
                    for r in result.rules
                ],
            }
        else:
            data = {
                "domain": result.domain,
                "source": {
                    "id": result.source.id,
                    "name": result.source.name,
                    "type": result.source.source_type,
                },
                "concepts": [
                    {
                        "id": c.id,
                        "name": c.name,
                        "description": c.description,
                    }
                    for c in result.concepts
                ],
                "edges": [
                    {
                        "source": e.source_id,
                        "target": e.target_id,
                        "relation": e.relation_type.value,
                    }
                    for e in result.edges
                ],
                "rules": [
                    {
                        "id": r.id,
                        "text": r.text,
                        "category": r.category,
                    }
                    for r in result.rules
                ],
            }

        return yaml.dump(data, default_flow_style=False, sort_keys=False)


class FileOutputStrategy(OutputStrategy[Path]):
    """Output to file (JSON or YAML based on extension)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    @property
    def name(self) -> str:
        return "file"

    def output(self, result: ExtractionResult) -> Path:
        """Write to file and return path."""
        ext = self.path.suffix.lower()

        if ext in (".json",):
            strategy = JSONOutputStrategy()
            content = strategy.output(result)
        elif ext in (".yaml", ".yml"):
            strategy = YAMLOutputStrategy()
            content = strategy.output(result)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(content)

        return self.path


class DirectKGOutputStrategy(OutputStrategy[int]):
    """Output directly to knowledge graph backend.

    Returns count of entities written.
    """

    def __init__(self, backend: Any):
        """
        Args:
            backend: GraphBackend instance to write to
        """
        self.backend = backend

    @property
    def name(self) -> str:
        return "kg"

    def output(self, result: ExtractionResult) -> int:
        """Write directly to KG and return entity count."""
        # Build manifest for ingestion
        manifest = IngestionManifest(
            source=result.source,
            domain=result.domain,
            concepts=result.concepts,
            edges=result.edges,
            rules=result.rules,
        )

        # Use backend's ingest method
        self.backend.ingest_manifest(manifest)

        return len(result.concepts) + len(result.edges) + len(result.rules)


# =============================================================================
# Auto-registration
# =============================================================================


def _ensure_registered() -> None:
    """Ensure default strategies are registered."""
    if _OUTPUT_STRATEGIES:
        return

    register_output_strategy("manifest", ManifestOutputStrategy)
    register_output_strategy("json", JSONOutputStrategy)
    register_output_strategy("yaml", YAMLOutputStrategy)
    register_output_strategy("file", FileOutputStrategy)
    register_output_strategy("kg", DirectKGOutputStrategy)
