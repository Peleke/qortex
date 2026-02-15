"""Pluggable text chunking for online session indexing.

Defines a ChunkingStrategy protocol and a default sentence-boundary
implementation. Strategies are swappable at runtime — pass any
callable matching the protocol to the ingest pipeline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Chunk:
    """Immutable text chunk with deterministic ID."""

    id: str
    text: str
    index: int


class ChunkingStrategy(Protocol):
    """Protocol for text chunking strategies.

    Any callable with this signature can be used as a chunker.
    Implementations: SentenceBoundaryChunker (default), or bring your own
    (tiktoken-based, semantic, LLM-assisted, etc.).
    """

    def __call__(
        self,
        text: str,
        max_tokens: int = 256,
        overlap_tokens: int = 32,
        source_id: str = "",
    ) -> list[Chunk]: ...


class SentenceBoundaryChunker:
    """Default chunking strategy: regex sentence-boundary splitting.

    Uses 1 token ~ 4 chars approximation. No external dependencies.
    """

    def __call__(
        self,
        text: str,
        max_tokens: int = 256,
        overlap_tokens: int = 32,
        source_id: str = "",
    ) -> list[Chunk]:
        if not text or not text.strip():
            return []

        max_chars = max_tokens * 4
        overlap_chars = overlap_tokens * 4

        sentences = re.split(r"(?<=[.!?\n])\s+", text.strip())
        if not sentences:
            return []

        chunks: list[Chunk] = []
        current = ""
        idx = 0

        for sentence in sentences:
            if len(current) + len(sentence) > max_chars and current:
                chunk_id = hashlib.sha256(
                    f"{source_id}:{idx}:{current[:64]}".encode()
                ).hexdigest()[:16]
                chunks.append(Chunk(id=chunk_id, text=current.strip(), index=idx))
                idx += 1
                current = current[-overlap_chars:] + " " + sentence if overlap_chars else sentence
            else:
                current = (current + " " + sentence).strip() if current else sentence

        if current.strip():
            chunk_id = hashlib.sha256(
                f"{source_id}:{idx}:{current[:64]}".encode()
            ).hexdigest()[:16]
            chunks.append(Chunk(id=chunk_id, text=current.strip(), index=idx))

        return chunks


# Default instance — use this unless you need a custom strategy.
default_chunker: ChunkingStrategy = SentenceBoundaryChunker()
