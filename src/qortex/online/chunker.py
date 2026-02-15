"""Lightweight text chunker for online message ingestion.

Splits text into overlapping chunks suitable for embedding. No LLM required.
Used by qortex_ingest_message for fast, non-blocking session indexing.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    index: int


def chunk_text(
    text: str,
    max_tokens: int = 256,
    overlap_tokens: int = 32,
    source_id: str = "",
) -> list[Chunk]:
    """Split text into overlapping chunks by sentence boundaries.

    Uses a simple sentence splitter (period/newline boundaries) with
    token approximation (1 token ~ 4 chars). No external dependencies.
    """
    if not text or not text.strip():
        return []

    # Approximate tokens as chars / 4
    max_chars = max_tokens * 4
    overlap_chars = overlap_tokens * 4

    # Split on sentence boundaries
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
            # Keep overlap from end of current chunk
            current = current[-overlap_chars:] + " " + sentence if overlap_chars else sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence

    # Final chunk
    if current.strip():
        chunk_id = hashlib.sha256(
            f"{source_id}:{idx}:{current[:64]}".encode()
        ).hexdigest()[:16]
        chunks.append(Chunk(id=chunk_id, text=current.strip(), index=idx))

    return chunks
