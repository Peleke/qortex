"""Tests for online session indexing: chunker, events, and MCP impl functions."""

from __future__ import annotations

import pytest
from qortex.observe.events import MessageIngested, ToolResultIngested

from qortex.online.chunker import Chunk, chunk_text

# ---------------------------------------------------------------------------
# Chunker tests
# ---------------------------------------------------------------------------


class TestChunkText:
    def test_empty_input_returns_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []
        assert chunk_text(None) == []  # type: ignore[arg-type]

    def test_short_text_single_chunk(self):
        chunks = chunk_text("Hello world.", max_tokens=256)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].index == 0

    def test_long_text_splits(self):
        # 256 tokens * 4 chars = 1024 chars max per chunk
        sentences = [f"This is sentence number {i}." for i in range(100)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_tokens=64)  # 256 chars max
        assert len(chunks) > 1
        # All text should be covered
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_chunk_ids_are_deterministic(self):
        text = "First sentence. Second sentence. Third sentence."
        c1 = chunk_text(text, source_id="test")
        c2 = chunk_text(text, source_id="test")
        assert [c.id for c in c1] == [c.id for c in c2]

    def test_different_source_ids_produce_different_chunk_ids(self):
        text = "Hello world. This is a test."
        c1 = chunk_text(text, source_id="a")
        c2 = chunk_text(text, source_id="b")
        assert c1[0].id != c2[0].id

    def test_chunk_indices_sequential(self):
        sentences = [f"Sentence {i}." for i in range(50)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_tokens=32)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_overlap_preserves_context(self):
        # With overlap, the end of chunk N should appear at the start of chunk N+1
        sentences = ["Word " * 60 + "." for _ in range(5)]
        text = " ".join(sentences)
        chunks = chunk_text(text, max_tokens=64, overlap_tokens=16)
        if len(chunks) >= 2:
            # Last 64 chars of chunk 0 should appear in chunk 1
            tail = chunks[0].text[-64:]
            assert tail in chunks[1].text

    def test_returns_chunk_dataclass(self):
        chunks = chunk_text("A simple test.")
        assert isinstance(chunks[0], Chunk)
        assert hasattr(chunks[0], "id")
        assert hasattr(chunks[0], "text")
        assert hasattr(chunks[0], "index")


# ---------------------------------------------------------------------------
# Event dataclass tests
# ---------------------------------------------------------------------------


class TestEvents:
    def test_message_ingested_frozen(self):
        evt = MessageIngested(
            session_id="s1", role="user", domain="test",
            chunk_count=3, concept_count=3, edge_count=2, latency_ms=12.5,
        )
        assert evt.session_id == "s1"
        assert evt.role == "user"
        with pytest.raises(AttributeError):
            evt.role = "assistant"  # type: ignore[misc]

    def test_tool_result_ingested_frozen(self):
        evt = ToolResultIngested(
            tool_name="search", session_id="s1", domain="test",
            concept_count=5, edge_count=4, latency_ms=8.3,
        )
        assert evt.tool_name == "search"
        with pytest.raises(AttributeError):
            evt.tool_name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Impl function tests (lightweight, mock-based)
# ---------------------------------------------------------------------------


class TestIngestMessageImpl:
    def test_empty_text_noop(self):
        from qortex.core.memory import InMemoryBackend
        from qortex.mcp.server import _ingest_message_impl, create_server

        backend = InMemoryBackend()
        backend.connect()
        create_server(backend=backend)

        result = _ingest_message_impl("", session_id="s1")
        assert result["chunks"] == 0
        assert result["concepts"] == 0

    def test_whitespace_only_noop(self):
        from qortex.core.memory import InMemoryBackend
        from qortex.mcp.server import _ingest_message_impl, create_server

        backend = InMemoryBackend()
        backend.connect()
        create_server(backend=backend)

        result = _ingest_message_impl("   \n  ", session_id="s1")
        assert result["chunks"] == 0


class TestIngestToolResultImpl:
    def test_empty_text_noop(self):
        from qortex.core.memory import InMemoryBackend
        from qortex.mcp.server import _ingest_tool_result_impl, create_server

        backend = InMemoryBackend()
        backend.connect()
        create_server(backend=backend)

        result = _ingest_tool_result_impl("search", "", session_id="s1")
        assert result["concepts"] == 0
        assert result["edges"] == 0
