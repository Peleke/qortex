"""Hippocampus: Cross-domain integration via GraphRAG-style retrieval.

The hippocampus is the cross-domain integrator:
- Maintains sparse index of key concepts across domains
- Builds bridges between related concepts in different domains
- Implements pattern completion for retrieval

Architecture models full GraphRAG, with graceful degradation:
- Full: PPR via MAGE, NER for concept extraction (GraphRAGAdapter)
- Fallback: Pure vector similarity (VecOnlyAdapter)
- Legacy: Simple BFS traversal, keyword matching (Hippocampus class)
"""

from .adapter import (
    GraphRAGAdapter,
    RetrievalAdapter,
    RetrievalItem,
    RetrievalResult,
    VecOnlyAdapter,
    get_adapter,
)
from .interoception import (
    InteroceptionConfig,
    InteroceptionProvider,
    LocalInteroceptionProvider,
    McpOutcomeSource,
    Outcome,
    OutcomeSource,
)
from .retrieval import Hippocampus

__all__ = [
    "Hippocampus",
    "GraphRAGAdapter",
    "InteroceptionConfig",
    "InteroceptionProvider",
    "LocalInteroceptionProvider",
    "McpOutcomeSource",
    "Outcome",
    "OutcomeSource",
    "RetrievalAdapter",
    "RetrievalItem",
    "RetrievalResult",
    "VecOnlyAdapter",
    "get_adapter",
]
