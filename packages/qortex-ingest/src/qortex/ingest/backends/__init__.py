"""Extraction backends for ingestion.

Backends implement LLMBackend protocol for concept/relation/rule extraction.
"""

from __future__ import annotations

import os
from typing import Literal

from qortex.ingest.base import LLMBackend, StubLLMBackend

BackendType = Literal["anthropic", "ollama", "stub"]


def get_extraction_backend(
    prefer: BackendType | None = None,
    model: str | None = None,
) -> LLMBackend:
    """Get an extraction backend, auto-detecting if no preference given.

    Args:
        prefer: Explicit backend choice. If None, auto-detect.
        model: Model name override (backend-specific).

    Auto-detection priority:
        1. Anthropic if ANTHROPIC_API_KEY is set
        2. Ollama if server is reachable at OLLAMA_HOST (default localhost:11434)
        3. StubLLMBackend (returns empty results, for testing)

    Raises:
        ValueError: If preferred backend is unavailable.
    """
    if prefer == "anthropic" or (prefer is None and os.environ.get("ANTHROPIC_API_KEY")):
        from qortex.ingest.backends.anthropic import AnthropicExtractionBackend

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            if prefer == "anthropic":
                raise ValueError("ANTHROPIC_API_KEY not set")
            # Fall through to next option
        else:
            return AnthropicExtractionBackend(api_key=api_key, model=model)

    if prefer == "ollama" or prefer is None:
        from qortex.ingest.backends.ollama import OllamaExtractionBackend

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        backend = OllamaExtractionBackend(host=host, model=model)

        if backend.is_available():
            return backend
        elif prefer == "ollama":
            raise ValueError(f"Ollama not reachable at {host}")

    if prefer == "stub" or prefer is None:
        return StubLLMBackend()

    raise ValueError(f"Unknown backend: {prefer}")


__all__ = [
    "get_extraction_backend",
    "BackendType",
]
