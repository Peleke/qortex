"""Composable ingestion pipeline using strategy pattern.

The Pipeline combines:
- InputStrategy: How to read and chunk the source
- LLMBackend: How to extract concepts/relations/rules
- OutputStrategy: How to format/write the output
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

from qortex.core.models import (
    ConceptEdge,
    ConceptNode,
    ExplicitRule,
    SourceMetadata,
)

from ..base import Source
from .input import InputStrategy, get_input_strategy
from .output import ExtractionResult, OutputStrategy, get_output_strategy

if TYPE_CHECKING:
    from ..llm import LLMBackend

T = TypeVar("T")


class Pipeline(Generic[T]):
    """Composable ingestion pipeline.

    Combines input, LLM, and output strategies into a single processor.

    Example:
        # Simple usage
        pipeline = Pipeline(
            input_strategy=get_input_strategy("markdown"),
            llm_backend=get_llm_backend(),
            output_strategy=get_output_strategy("json"),
        )
        json_output = pipeline.process(source)

        # Direct to KG
        pipeline = Pipeline(
            input_strategy=get_input_strategy("pdf"),
            llm_backend=get_llm_backend("anthropic"),
            output_strategy=get_output_strategy("kg", backend=memgraph),
        )
        count = pipeline.process(source)
    """

    def __init__(
        self,
        input_strategy: InputStrategy,
        llm_backend: "LLMBackend",
        output_strategy: OutputStrategy[T],
    ):
        self.input_strategy = input_strategy
        self.llm_backend = llm_backend
        self.output_strategy = output_strategy

    def process(
        self,
        source: Source,
        domain: str | None = None,
    ) -> T:
        """Run the full ingestion pipeline.

        Args:
            source: Input source to process
            domain: Domain name (auto-detected if None)

        Returns:
            Output as determined by output_strategy type
        """
        # 1. Chunk the source
        chunks = self.input_strategy.chunk(source)

        if not chunks:
            raise ValueError("Source produced no chunks")

        # 2. Determine domain
        if domain is None:
            sample = chunks[0].content[:1000]
            domain = self.llm_backend.suggest_domain_name(
                source.name or "unknown",
                sample,
            )

        # 3. Extract concepts
        source_id = f"{domain}:{source.name or 'source'}"
        concepts: list[ConceptNode] = []
        seen_concepts: set[str] = set()

        for chunk in chunks:
            extracted = self.llm_backend.extract_concepts(chunk.content, domain)
            for c in extracted:
                # Deduplicate by name
                if c["name"].lower() in seen_concepts:
                    continue
                seen_concepts.add(c["name"].lower())

                concepts.append(ConceptNode(
                    id=f"{domain}:{c['name']}",
                    name=c["name"],
                    description=c.get("description", ""),
                    domain=domain,
                    source_id=source_id,
                    source_location=chunk.location,
                    confidence=c.get("confidence", 1.0),
                ))

        # 4. Extract relations
        all_text = "\n\n".join(c.content for c in chunks)
        relation_dicts = self.llm_backend.extract_relations(concepts, all_text)
        edges = [
            ConceptEdge(
                source_id=r["source_id"],
                target_id=r["target_id"],
                relation_type=r["relation_type"],
                confidence=r.get("confidence", 1.0),
            )
            for r in relation_dicts
        ]

        # 5. Extract rules
        rule_dicts = self.llm_backend.extract_rules(all_text, concepts)
        rules = [
            ExplicitRule(
                id=f"{domain}:rule:{i}",
                text=r["text"],
                domain=domain,
                source_id=source_id,
                concept_ids=r.get("concept_ids", []),
                source_location=None,
                category=r.get("category"),
                confidence=r.get("confidence", 1.0),
            )
            for i, r in enumerate(rule_dicts)
        ]

        # 6. Build extraction result
        source_meta = SourceMetadata(
            id=source_id,
            name=source.name or "unknown",
            source_type=source.source_type,
            path_or_url=str(source.path or source.url or "raw"),
            chunk_count=len(chunks),
            concept_count=len(concepts),
            rule_count=len(rules),
        )

        result = ExtractionResult(
            source=source_meta,
            domain=domain,
            concepts=concepts,
            edges=edges,
            rules=rules,
        )

        # 7. Output
        return self.output_strategy.output(result)

    @classmethod
    def auto(
        cls,
        source: Source,
        llm_backend: "LLMBackend",
        output: str = "manifest",
        **output_kwargs,
    ) -> "Pipeline[Any]":
        """Create pipeline with auto-detected input strategy.

        Args:
            source: Source to process (used to detect input strategy)
            llm_backend: LLM backend to use
            output: Output strategy name
            **output_kwargs: Additional args for output strategy

        Returns:
            Configured Pipeline
        """
        input_strategy = get_input_strategy(source=source)
        output_strategy = get_output_strategy(output, **output_kwargs)

        return cls(
            input_strategy=input_strategy,
            llm_backend=llm_backend,
            output_strategy=output_strategy,
        )


# =============================================================================
# Convenience functions
# =============================================================================


def ingest(
    source: Source,
    llm_backend: "LLMBackend",
    domain: str | None = None,
    output: str = "manifest",
    **kwargs,
) -> Any:
    """One-shot ingestion with auto-detection.

    Args:
        source: Input source
        llm_backend: LLM backend for extraction
        domain: Domain name (auto-detected if None)
        output: Output strategy ("manifest", "json", "yaml", "kg")
        **kwargs: Additional args for output strategy

    Returns:
        Extraction result in requested format
    """
    pipeline = Pipeline.auto(source, llm_backend, output, **kwargs)
    return pipeline.process(source, domain)
