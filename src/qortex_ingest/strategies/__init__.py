"""Strategy system for pluggable input/output handling.

Input strategies handle different source formats:
- TextInputStrategy: Plain text
- MarkdownInputStrategy: Markdown documents
- PDFInputStrategy: PDF files
- URLInputStrategy: Web content (future)

Output strategies handle different targets:
- ManifestOutputStrategy: IngestionManifest (default)
- DirectKGOutputStrategy: Write directly to graph backend
- JSONOutputStrategy: Export as JSON
- YAMLOutputStrategy: Export as YAML (buildlog-compatible)

Usage:
    from qortex_ingest.strategies import (
        get_input_strategy,
        get_output_strategy,
        Pipeline,
    )

    # Build a pipeline
    pipeline = Pipeline(
        input_strategy=get_input_strategy("markdown"),
        llm_backend=get_llm_backend(),
        output_strategy=get_output_strategy("manifest"),
    )

    # Run it
    result = pipeline.process(source)
"""

from .input import (
    InputStrategy,
    get_input_strategy,
    register_input_strategy,
    available_input_strategies,
)
from .output import (
    OutputStrategy,
    get_output_strategy,
    register_output_strategy,
    available_output_strategies,
)
from .pipeline import Pipeline

__all__ = [
    # Input
    "InputStrategy",
    "get_input_strategy",
    "register_input_strategy",
    "available_input_strategies",
    # Output
    "OutputStrategy",
    "get_output_strategy",
    "register_output_strategy",
    "available_output_strategies",
    # Pipeline
    "Pipeline",
]
