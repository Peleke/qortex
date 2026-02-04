"""LLM backend system with swappable strategies.

Usage:
    from qortex_ingest.llm import get_llm_backend, LLMBackend

    # Auto-select best available backend
    llm = get_llm_backend()

    # Or explicitly choose
    llm = get_llm_backend("anthropic")
    llm = get_llm_backend("ollama", model="llama3")
    llm = get_llm_backend("keyword")  # No LLM, keyword extraction only
"""

from .base import LLMBackend, LLMConfig
from .registry import get_llm_backend, register_backend, available_backends

__all__ = [
    "LLMBackend",
    "LLMConfig",
    "get_llm_backend",
    "register_backend",
    "available_backends",
]
