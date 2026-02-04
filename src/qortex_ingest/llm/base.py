"""Base LLM backend protocol and configuration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from qortex.core.models import ConceptNode


@dataclass
class LLMConfig:
    """Configuration for LLM backends."""

    # Model selection
    model: str | None = None  # Backend-specific model name

    # API configuration
    api_key: str | None = None
    base_url: str | None = None

    # Generation parameters
    temperature: float = 0.3  # Lower for more deterministic extraction
    max_tokens: int = 4096

    # Extraction parameters
    max_concepts: int = 20
    max_rules: int = 15
    min_confidence: float = 0.5

    # Additional backend-specific options
    options: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for LLM backends.

    All backends must implement these methods for concept/relation extraction.
    The protocol is designed to be LLM-agnostic.
    """

    @property
    def name(self) -> str:
        """Backend identifier (e.g., 'anthropic', 'ollama', 'keyword')."""
        ...

    @property
    def is_available(self) -> bool:
        """Whether this backend is currently available (API key set, service running, etc.)."""
        ...

    def extract_concepts(self, text: str, domain_hint: str | None = None) -> list[dict]:
        """Extract concepts from text.

        Returns list of dicts with keys:
        - name: str
        - description: str
        - confidence: float (0-1)
        """
        ...

    def extract_relations(
        self,
        concepts: list[ConceptNode],
        text: str,
    ) -> list[dict]:
        """Extract relations between concepts.

        Returns list of dicts with keys:
        - source_id: str
        - target_id: str
        - relation_type: RelationType
        - confidence: float (0-1)
        """
        ...

    def extract_rules(self, text: str, concepts: list[ConceptNode]) -> list[dict]:
        """Extract explicit rules from text.

        Returns list of dicts with keys:
        - text: str
        - concept_ids: list[str]
        - category: str | None
        - confidence: float (0-1)
        """
        ...

    def suggest_domain_name(self, source_name: str, sample_text: str) -> str:
        """Suggest a domain name for a source."""
        ...


class BaseLLMBackend(ABC):
    """Base class for LLM backends with common functionality."""

    def __init__(self, config: LLMConfig | None = None):
        self.config = config or LLMConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this backend is available."""
        ...

    @abstractmethod
    def extract_concepts(self, text: str, domain_hint: str | None = None) -> list[dict]:
        ...

    @abstractmethod
    def extract_relations(self, concepts: list[ConceptNode], text: str) -> list[dict]:
        ...

    @abstractmethod
    def extract_rules(self, text: str, concepts: list[ConceptNode]) -> list[dict]:
        ...

    @abstractmethod
    def suggest_domain_name(self, source_name: str, sample_text: str) -> str:
        ...
