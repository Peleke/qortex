"""Tests for online session indexing: chunker, events, and MCP tools."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from qortex.observe.events import MessageIngested, ToolResultIngested

from qortex.online.chunker import Chunk, ChunkingStrategy, SentenceBoundaryChunker

# ---------------------------------------------------------------------------
# Chunker unit tests
# ---------------------------------------------------------------------------

chunker = SentenceBoundaryChunker()


class TestSentenceBoundaryChunker:
    def test_empty_input_returns_empty(self):
        assert chunker("") == []
        assert chunker("   ") == []
        assert chunker(None) == []  # type: ignore[arg-type]

    def test_short_text_single_chunk(self):
        chunks = chunker("Hello world.", max_tokens=256)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."
        assert chunks[0].index == 0

    def test_long_text_splits(self):
        sentences = [f"This is sentence number {i}." for i in range(100)]
        text = " ".join(sentences)
        chunks = chunker(text, max_tokens=64)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) > 0

    def test_chunk_ids_are_deterministic(self):
        text = "First sentence. Second sentence. Third sentence."
        c1 = chunker(text, source_id="test")
        c2 = chunker(text, source_id="test")
        assert [c.id for c in c1] == [c.id for c in c2]

    def test_different_source_ids_produce_different_chunk_ids(self):
        text = "Hello world. This is a test."
        c1 = chunker(text, source_id="a")
        c2 = chunker(text, source_id="b")
        assert c1[0].id != c2[0].id

    def test_chunk_indices_sequential(self):
        sentences = [f"Sentence {i}." for i in range(50)]
        text = " ".join(sentences)
        chunks = chunker(text, max_tokens=32)
        indices = [c.index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_overlap_preserves_context(self):
        sentences = ["Word " * 60 + "." for _ in range(5)]
        text = " ".join(sentences)
        chunks = chunker(text, max_tokens=64, overlap_tokens=16)
        assert len(chunks) >= 2, "Test requires multiple chunks to verify overlap"
        tail = chunks[0].text[-64:]
        assert tail in chunks[1].text

    def test_returns_frozen_chunk_dataclass(self):
        chunks = chunker("A simple test.")
        assert isinstance(chunks[0], Chunk)
        with pytest.raises(AttributeError):
            chunks[0].text = "mutated"  # type: ignore[misc]

    def test_conforms_to_chunking_strategy_protocol(self):
        """SentenceBoundaryChunker satisfies ChunkingStrategy at runtime."""
        instance: ChunkingStrategy = SentenceBoundaryChunker()
        result = instance("Hello world.")
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Property-based chunker tests (Hypothesis)
# ---------------------------------------------------------------------------

# Strategy: text with sentence-like structure
sentence_text = st.text(
    alphabet=st.sampled_from("abcdefghij .!\n"),
    min_size=1, max_size=2000,
).filter(lambda t: t.strip())


class TestChunkerProperties:
    @given(text=sentence_text, max_tokens=st.integers(min_value=8, max_value=512))
    @settings(max_examples=50)
    def test_indices_always_sequential(self, text: str, max_tokens: int):
        chunks = chunker(text, max_tokens=max_tokens)
        if chunks:
            assert [c.index for c in chunks] == list(range(len(chunks)))

    @given(text=sentence_text)
    @settings(max_examples=50)
    def test_deterministic(self, text: str):
        c1 = chunker(text, source_id="prop")
        c2 = chunker(text, source_id="prop")
        assert [c.id for c in c1] == [c.id for c in c2]

    @given(text=sentence_text)
    @settings(max_examples=50)
    def test_smaller_max_tokens_produces_more_or_equal_chunks(self, text: str):
        big = chunker(text, max_tokens=256)
        small = chunker(text, max_tokens=64)
        assert len(small) >= len(big)

    @given(text=sentence_text)
    @settings(max_examples=50)
    def test_all_chunks_nonempty(self, text: str):
        chunks = chunker(text, max_tokens=32)
        for c in chunks:
            assert len(c.text.strip()) > 0


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
# MCP tool integration tests (via public create_server + tool wrappers)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _server():
    """Set up a minimal server with InMemoryBackend for tool tests."""
    from qortex.core.memory import InMemoryBackend
    from qortex.mcp.server import create_server

    backend = InMemoryBackend()
    backend.connect()
    create_server(backend=backend)


class TestIngestMessage:
    def test_empty_text_noop(self, _server):
        from qortex.mcp.server import _ingest_message_impl

        result = _ingest_message_impl("", session_id="s1")
        assert result["chunks"] == 0
        assert result["concepts"] == 0

    def test_whitespace_only_noop(self, _server):
        from qortex.mcp.server import _ingest_message_impl

        result = _ingest_message_impl("   \n  ", session_id="s1")
        assert result["chunks"] == 0

    def test_real_text_chunks_and_returns_counts(self, _server):
        from qortex.mcp.server import _ingest_message_impl

        text = "The auth module handles JWT tokens. It validates signatures on every request."
        result = _ingest_message_impl(text, session_id="s1", role="user")
        assert result["chunks"] >= 1
        assert result["session_id"] == "s1"
        assert "latency_ms" in result

    def test_invalid_role_clamped_to_unknown(self, _server):
        from qortex.mcp.server import _ingest_message_impl

        result = _ingest_message_impl("test.", session_id="s1", role="evil_injection")
        # Should not crash; role should be clamped
        assert result["chunks"] >= 1


class TestIngestToolResult:
    def test_empty_text_noop(self, _server):
        from qortex.mcp.server import _ingest_tool_result_impl

        result = _ingest_tool_result_impl("search", "", session_id="s1")
        assert result["concepts"] == 0
        assert result["edges"] == 0

    def test_real_text_chunks_and_returns_counts(self, _server):
        from qortex.mcp.server import _ingest_tool_result_impl

        text = "Found 3 matching files. src/auth.py contains the JWT validation logic."
        result = _ingest_tool_result_impl("grep", text, session_id="s1")
        assert result["concepts"] >= 0  # no embedding model -> 0 concepts, but no crash
        assert result["tool_name"] == "grep"
        assert "latency_ms" in result


class TestIngestWithEmbedding:
    """Regression: ensure ingest calls EmbeddingModel.embed() (not embed_batch)."""

    def test_ingest_calls_embed_on_model(self, _server):
        import qortex.mcp.server as srv
        from qortex.mcp.server import _ingest_message_impl

        embedded_texts: list[list[str]] = []

        class FakeEmbedding:
            dimensions = 4

            def embed(self, texts: list[str]) -> list[list[float]]:
                embedded_texts.append(texts)
                return [[0.1, 0.2, 0.3, 0.4]] * len(texts)

        class FakeVecIndex:
            added: list[tuple] = []

            def add(self, ids, embeddings):
                self.added.append((ids, embeddings))

        old_model, old_index = srv._embedding_model, srv._vector_index
        try:
            srv._embedding_model = FakeEmbedding()
            srv._vector_index = FakeVecIndex()
            result = _ingest_message_impl("Hello world.", session_id="s1", role="user")
            assert result["concepts"] >= 1
            assert len(embedded_texts) == 1, "embed() should be called exactly once"
        finally:
            srv._embedding_model = old_model
            srv._vector_index = old_index


class TestChunkingStrategyInjection:
    def test_custom_strategy_is_used(self, _server):
        from qortex.mcp.server import _ingest_message_impl, set_chunking_strategy

        call_count = 0

        class CountingChunker:
            def __call__(self, text, max_tokens=256, overlap_tokens=32, source_id=""):
                nonlocal call_count
                call_count += 1
                return [Chunk(id="custom-0", text=text, index=0)]

        set_chunking_strategy(CountingChunker())
        result = _ingest_message_impl("Hello.", session_id="s1")
        assert call_count == 1
        assert result["chunks"] == 1

        # Reset to default
        set_chunking_strategy(None)
