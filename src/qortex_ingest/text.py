"""Text ingestor - simplest format, LLM-assisted chunking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Chunk, Ingestor, LLMBackend, Source

if TYPE_CHECKING:
    from qortex.core.pruning import PruningConfig


class TextIngestor(Ingestor):
    """Ingest plain text sources.

    Uses LLM to identify natural chunk boundaries.
    Simplest ingestor - good starting point.
    """

    def __init__(
        self,
        llm: LLMBackend,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        pruning_config: PruningConfig | None = None,
    ):
        super().__init__(llm, pruning_config=pruning_config)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, source: Source) -> list[Chunk]:
        """Chunk text by size with overlap.

        TODO: Use LLM to identify semantic boundaries instead of fixed size.
        """
        if source.raw_content:
            content = source.raw_content
        elif source.path:
            content = source.path.read_text()
        else:
            raise ValueError("Text source must have raw_content or path")

        chunks = []
        start = 0
        chunk_num = 0

        while start < len(content):
            end = start + self.chunk_size
            chunk_text = content[start:end]

            # Try to break at paragraph boundary
            if end < len(content):
                last_para = chunk_text.rfind("\n\n")
                if last_para > self.chunk_size // 2:
                    chunk_text = chunk_text[:last_para]
                    end = start + last_para

            chunks.append(Chunk(
                id=f"chunk_{chunk_num}",
                content=chunk_text.strip(),
                location=f"chars {start}-{end}",
            ))

            start = end - self.chunk_overlap
            chunk_num += 1

        return chunks
