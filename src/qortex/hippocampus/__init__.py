"""Hippocampus: Cross-domain integration via HippoRAG-style retrieval.

The hippocampus is the cross-domain integrator:
- Maintains sparse index of key concepts across domains
- Builds bridges between related concepts in different domains
- Implements pattern completion for retrieval

Architecture models full HippoRAG, with graceful degradation:
- Full: PPR via MAGE, NER for concept extraction
- Fallback: Simple BFS traversal, keyword matching
"""

from .retrieval import Hippocampus

__all__ = ["Hippocampus"]
