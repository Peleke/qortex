"""qortex_ingest: Pluggable ingestion system for qortex.

This package is SEPARABLE from qortex core.
It produces IngestionManifest objects that the KG consumes.

## Architecture

Uses strategy pattern throughout:
- InputStrategy: How to read/chunk sources (text, markdown, pdf)
- LLMBackend: How to extract concepts/relations (anthropic, ollama, keyword)
- OutputStrategy: How to format/write results (manifest, json, yaml, kg)

## Usage

    from qortex_ingest import Pipeline, Source
    from qortex_ingest.llm import get_llm_backend
    from qortex_ingest.strategies import get_input_strategy, get_output_strategy

    # Build a pipeline
    pipeline = Pipeline(
        input_strategy=get_input_strategy("markdown"),
        llm_backend=get_llm_backend(),  # Auto-selects best available
        output_strategy=get_output_strategy("json"),
    )

    # Process a source
    source = Source(raw_content="...", source_type="markdown", name="my_doc")
    json_output = pipeline.process(source)

    # Or use the convenience function
    from qortex_ingest.strategies import ingest

    result = ingest(source, get_llm_backend(), output="yaml")

Could become its own package later:
- qortex: KG + hippocampus + projectors
- qortex-ingest: Ingestors + LLM backends + strategies
"""

from qortex.core.models import IngestionManifest

from .base import Chunk, Source
from .strategies import Pipeline
from .strategies.pipeline import ingest

__all__ = [
    # Models
    "IngestionManifest",
    "Source",
    "Chunk",
    # Pipeline
    "Pipeline",
    "ingest",
]
