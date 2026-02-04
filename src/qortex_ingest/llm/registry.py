"""LLM backend registry and factory.

Provides:
- Auto-detection of available backends
- Priority-based selection
- Easy backend switching
"""

from __future__ import annotations

import os
from typing import Callable, Type

from .base import BaseLLMBackend, LLMBackend, LLMConfig

# Registry of backend classes
_BACKENDS: dict[str, Type[BaseLLMBackend]] = {}

# Priority order for auto-selection (higher = preferred)
_PRIORITY: dict[str, int] = {
    "anthropic": 100,  # Best quality
    "ollama": 50,      # Good for local/private
    "keyword": 1,      # Always available fallback
}


def register_backend(
    name: str,
    backend_class: Type[BaseLLMBackend],
    priority: int | None = None,
) -> None:
    """Register a backend class.

    Args:
        name: Backend identifier
        backend_class: Class implementing BaseLLMBackend
        priority: Selection priority (higher = preferred)
    """
    _BACKENDS[name] = backend_class
    if priority is not None:
        _PRIORITY[name] = priority


def available_backends() -> list[str]:
    """List backends that are currently available.

    Returns:
        List of backend names sorted by priority
    """
    _ensure_registered()

    available = []
    for name, cls in _BACKENDS.items():
        try:
            instance = cls()
            if instance.is_available:
                available.append(name)
        except Exception:
            pass

    # Sort by priority (descending)
    return sorted(available, key=lambda n: _PRIORITY.get(n, 0), reverse=True)


def get_llm_backend(
    backend: str | None = None,
    config: LLMConfig | None = None,
    **kwargs,
) -> LLMBackend:
    """Get an LLM backend instance.

    Args:
        backend: Backend name ("anthropic", "ollama", "keyword") or None for auto-select
        config: LLM configuration
        **kwargs: Additional config overrides (api_key, model, etc.)

    Returns:
        Configured LLM backend instance

    Examples:
        # Auto-select best available
        llm = get_llm_backend()

        # Explicit selection
        llm = get_llm_backend("anthropic", api_key="sk-...")
        llm = get_llm_backend("ollama", model="llama3")
        llm = get_llm_backend("keyword")  # No LLM, keyword extraction
    """
    _ensure_registered()

    # Build config
    if config is None:
        config = LLMConfig()

    # Apply kwargs to config
    if kwargs:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    # Auto-select if no backend specified
    if backend is None:
        available = available_backends()
        if not available:
            raise RuntimeError("No LLM backends available")
        backend = available[0]

    # Get backend class
    if backend not in _BACKENDS:
        raise ValueError(f"Unknown backend: {backend}. Available: {list(_BACKENDS.keys())}")

    cls = _BACKENDS[backend]
    instance = cls(config)

    if not instance.is_available:
        # Try to provide helpful error message
        if backend == "anthropic":
            raise RuntimeError(
                "Anthropic backend not available. "
                "Set ANTHROPIC_API_KEY environment variable or pass api_key parameter."
            )
        elif backend == "ollama":
            raise RuntimeError(
                "Ollama backend not available. "
                "Ensure Ollama is running: `ollama serve`"
            )
        else:
            raise RuntimeError(f"Backend '{backend}' is not available")

    return instance


def _ensure_registered() -> None:
    """Ensure default backends are registered."""
    if _BACKENDS:
        return

    # Import and register default backends
    from .keyword import KeywordLLMBackend
    from .anthropic import AnthropicLLMBackend
    from .ollama import OllamaLLMBackend

    register_backend("keyword", KeywordLLMBackend, priority=1)
    register_backend("anthropic", AnthropicLLMBackend, priority=100)
    register_backend("ollama", OllamaLLMBackend, priority=50)


# Convenience function to check what's available
def get_best_available() -> str | None:
    """Get the name of the best available backend, or None if none available."""
    available = available_backends()
    return available[0] if available else None


def has_anthropic() -> bool:
    """Check if Anthropic (Claude) is available."""
    return "anthropic" in available_backends()


def has_ollama() -> bool:
    """Check if Ollama is available."""
    return "ollama" in available_backends()
