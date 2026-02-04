"""Input strategies for different source formats.

Each strategy knows how to:
1. Validate if it can handle a source
2. Chunk the source into processable units
3. Provide metadata about the source
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Type

from ..base import Chunk, Source


@dataclass
class SourceMetrics:
    """Metrics about processed source."""
    chunk_count: int
    total_chars: int
    estimated_tokens: int  # Rough estimate: chars / 4


class InputStrategy(ABC):
    """Base class for input format strategies.

    Subclasses implement format-specific chunking and validation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier (e.g., 'text', 'markdown', 'pdf')."""
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """File extensions this strategy handles (e.g., ['.md', '.markdown'])."""
        ...

    @abstractmethod
    def can_handle(self, source: Source) -> bool:
        """Check if this strategy can process the given source."""
        ...

    @abstractmethod
    def chunk(self, source: Source) -> list[Chunk]:
        """Split source into chunks for processing."""
        ...

    def get_content(self, source: Source) -> str:
        """Extract text content from source."""
        if source.raw_content:
            return source.raw_content
        elif source.path and source.path.exists():
            return source.path.read_text()
        else:
            raise ValueError(f"Cannot extract content from source: {source}")

    def metrics(self, chunks: list[Chunk]) -> SourceMetrics:
        """Compute metrics about chunked source."""
        total_chars = sum(len(c.content) for c in chunks)
        return SourceMetrics(
            chunk_count=len(chunks),
            total_chars=total_chars,
            estimated_tokens=total_chars // 4,
        )


# =============================================================================
# Registry
# =============================================================================

_INPUT_STRATEGIES: dict[str, Type[InputStrategy]] = {}


def register_input_strategy(name: str, strategy_class: Type[InputStrategy]) -> None:
    """Register an input strategy."""
    _INPUT_STRATEGIES[name] = strategy_class


def available_input_strategies() -> list[str]:
    """List registered input strategies."""
    _ensure_registered()
    return list(_INPUT_STRATEGIES.keys())


def get_input_strategy(
    name: str | None = None,
    source: Source | None = None,
    **kwargs,
) -> InputStrategy:
    """Get an input strategy.

    Args:
        name: Strategy name (e.g., 'text', 'markdown', 'pdf')
        source: Source to auto-detect strategy for (if name not provided)
        **kwargs: Strategy-specific configuration

    Returns:
        Configured InputStrategy instance
    """
    _ensure_registered()

    # Auto-detect from source
    if name is None and source is not None:
        for strategy_name, cls in _INPUT_STRATEGIES.items():
            instance = cls(**kwargs)
            if instance.can_handle(source):
                return instance
        raise ValueError(f"No strategy can handle source: {source}")

    # Explicit selection
    if name is None:
        name = "text"  # Default

    if name not in _INPUT_STRATEGIES:
        raise ValueError(f"Unknown input strategy: {name}. Available: {list(_INPUT_STRATEGIES.keys())}")

    return _INPUT_STRATEGIES[name](**kwargs)


# =============================================================================
# Built-in Strategies
# =============================================================================


class TextInputStrategy(InputStrategy):
    """Plain text chunking strategy."""

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def name(self) -> str:
        return "text"

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt", ".text", ""]

    def can_handle(self, source: Source) -> bool:
        if source.source_type == "text":
            return True
        if source.raw_content:
            return True
        if source.path:
            return source.path.suffix.lower() in self.supported_extensions
        return False

    def chunk(self, source: Source) -> list[Chunk]:
        """Chunk text by size with overlap."""
        content = self.get_content(source)
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


class MarkdownInputStrategy(InputStrategy):
    """Markdown chunking strategy - chunks by headings."""

    @property
    def name(self) -> str:
        return "markdown"

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown", ".mdx"]

    def can_handle(self, source: Source) -> bool:
        if source.source_type == "markdown":
            return True
        if source.path:
            return source.path.suffix.lower() in self.supported_extensions
        # Check content for markdown patterns
        if source.raw_content:
            return bool(re.search(r'^#{1,6}\s+', source.raw_content, re.MULTILINE))
        return False

    def chunk(self, source: Source) -> list[Chunk]:
        """Chunk markdown by headings."""
        content = self.get_content(source)
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

            level = len(match.group(1))
            title = match.group(2).strip()

            # Update hierarchy
            while len(current_hierarchy) >= level:
                current_hierarchy.pop()
            current_hierarchy.append(title)

            location = " > ".join(current_hierarchy)

            chunks.append(Chunk(
                id=f"section_{len(chunks)}",
                content=f"# {title}",
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


class PDFInputStrategy(InputStrategy):
    """PDF chunking strategy."""

    @property
    def name(self) -> str:
        return "pdf"

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def can_handle(self, source: Source) -> bool:
        if source.source_type == "pdf":
            return True
        if source.path:
            return source.path.suffix.lower() == ".pdf"
        return False

    def chunk(self, source: Source) -> list[Chunk]:
        """Extract and chunk PDF content."""
        if not source.path:
            raise ValueError("PDF source must have a path")

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PDF support requires PyMuPDF. Install with: pip install qortex[pdf]"
            )

        doc = fitz.open(source.path)
        chunks = []

        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                chunks.append(Chunk(
                    id=f"page_{page_num}",
                    content=text,
                    location=f"Page {page_num + 1}",
                    page=page_num + 1,
                ))

        return chunks


# =============================================================================
# Auto-registration
# =============================================================================


def _ensure_registered() -> None:
    """Ensure default strategies are registered."""
    if _INPUT_STRATEGIES:
        return

    # Order matters for auto-detection: more specific first
    register_input_strategy("pdf", PDFInputStrategy)
    register_input_strategy("markdown", MarkdownInputStrategy)
    register_input_strategy("text", TextInputStrategy)  # Most generic, last
