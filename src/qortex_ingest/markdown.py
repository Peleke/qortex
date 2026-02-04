"""Markdown ingestor - chunk by headings."""

from __future__ import annotations

import re

from .base import Chunk, Ingestor, Source


class MarkdownIngestor(Ingestor):
    """Ingest Markdown sources.

    Chunks by heading structure, preserving hierarchy.
    """

    def chunk(self, source: Source) -> list[Chunk]:
        """Chunk markdown by headings."""
        if source.raw_content:
            content = source.raw_content
        elif source.path:
            content = source.path.read_text()
        else:
            raise ValueError("Markdown source must have raw_content or path")

        # Split by headings
        heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

        chunks = []
        last_end = 0
        current_hierarchy: list[str] = []

        for match in heading_pattern.finditer(content):
            # Content before this heading
            if match.start() > last_end:
                pre_content = content[last_end:match.start()].strip()
                if pre_content and chunks:
                    # Append to previous chunk
                    chunks[-1] = Chunk(
                        id=chunks[-1].id,
                        content=chunks[-1].content + "\n\n" + pre_content,
                        location=chunks[-1].location,
                        level=chunks[-1].level,
                    )

            level = len(match.group(1))  # Number of #s
            title = match.group(2).strip()

            # Update hierarchy
            while len(current_hierarchy) >= level:
                current_hierarchy.pop()
            current_hierarchy.append(title)

            location = " > ".join(current_hierarchy)

            chunks.append(Chunk(
                id=f"section_{len(chunks)}",
                content=f"# {title}",  # Start with heading
                location=location,
                level=level,
            ))

            last_end = match.end()

        # Remaining content
        if last_end < len(content):
            remaining = content[last_end:].strip()
            if remaining:
                if chunks:
                    chunks[-1] = Chunk(
                        id=chunks[-1].id,
                        content=chunks[-1].content + "\n\n" + remaining,
                        location=chunks[-1].location,
                        level=chunks[-1].level,
                    )
                else:
                    chunks.append(Chunk(
                        id="section_0",
                        content=remaining,
                        location="root",
                        level=0,
                    ))

        return chunks
