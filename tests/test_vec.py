"""Tests for the qortex vector layer."""

import math

import pytest

from qortex.vec.index import NumpyVectorIndex


# =============================================================================
# NumpyVectorIndex
# =============================================================================


class TestNumpyVectorIndex:
    """Tests for NumpyVectorIndex (in-memory, brute-force cosine sim)."""

    def test_add_and_size(self):
        idx = NumpyVectorIndex(dimensions=3)
        assert idx.size() == 0
        idx.add(["a", "b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert idx.size() == 2

    def test_search_returns_correct_order(self):
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(
            ["x", "y", "z"],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]],
        )
        results = idx.search([1.0, 0.0, 0.0], top_k=3)
        # "x" should be most similar (exact match), then "z" (partial), then "y" (orthogonal)
        assert results[0][0] == "x"
        assert results[0][1] == pytest.approx(1.0, abs=0.01)
        assert results[1][0] == "z"
        assert results[2][0] == "y"

    def test_search_threshold_filters(self):
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["a", "b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        results = idx.search([1.0, 0.0, 0.0], top_k=10, threshold=0.5)
        assert len(results) == 1
        assert results[0][0] == "a"

    def test_search_top_k_limits(self):
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(
            ["a", "b", "c"],
            [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.8, 0.2, 0.0]],
        )
        results = idx.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_remove(self):
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["a", "b", "c"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        idx.remove(["b"])
        assert idx.size() == 2
        results = idx.search([0, 1, 0], top_k=10)
        ids = [r[0] for r in results]
        assert "b" not in ids

    def test_upsert_semantics(self):
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["a"], [[1, 0, 0]])
        idx.add(["a"], [[0, 1, 0]])  # overwrite
        assert idx.size() == 1
        results = idx.search([0, 1, 0], top_k=1)
        assert results[0][0] == "a"
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_empty_search(self):
        idx = NumpyVectorIndex(dimensions=3)
        results = idx.search([1, 0, 0])
        assert results == []

    def test_zero_vector_search(self):
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["a"], [[1, 0, 0]])
        results = idx.search([0, 0, 0])
        assert results == []

    def test_dimension_mismatch_raises(self):
        idx = NumpyVectorIndex(dimensions=3)
        with pytest.raises(ValueError, match="dimensions"):
            idx.add(["a"], [[1, 0]])

    def test_ids_embeddings_length_mismatch(self):
        idx = NumpyVectorIndex(dimensions=3)
        with pytest.raises(ValueError, match="must match"):
            idx.add(["a", "b"], [[1, 0, 0]])

    def test_cosine_similarity_correctness(self):
        """Verify cosine similarity matches manual computation."""
        idx = NumpyVectorIndex(dimensions=2)
        # 45-degree angle = cos(45) ≈ 0.707
        idx.add(["a"], [[1.0, 0.0]])
        results = idx.search([1.0, 1.0], top_k=1)
        expected = 1.0 / math.sqrt(2)
        assert results[0][1] == pytest.approx(expected, abs=0.001)

    def test_persist_is_noop(self):
        idx = NumpyVectorIndex(dimensions=3)
        idx.persist()  # Should not raise


# =============================================================================
# InMemoryBackend vector integration
# =============================================================================


class TestInMemoryBackendVectorSearch:
    """Tests for vector search via InMemoryBackend."""

    def test_add_and_search_embedding(self):
        from qortex.core.memory import InMemoryBackend
        from qortex.core.models import ConceptNode

        backend = InMemoryBackend()
        backend.connect()
        backend.create_domain("test")

        node = ConceptNode(
            id="test:foo", name="foo", description="A foo concept",
            domain="test", source_id="s1",
        )
        backend.add_node(node)
        backend.add_embedding("test:foo", [1.0, 0.0, 0.0])

        results = backend.vector_search([1.0, 0.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0][0].id == "test:foo"
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_vector_search_with_domain_filter(self):
        from qortex.core.memory import InMemoryBackend
        from qortex.core.models import ConceptNode

        backend = InMemoryBackend()
        backend.connect()
        backend.create_domain("d1")
        backend.create_domain("d2")

        n1 = ConceptNode(id="d1:a", name="a", description="", domain="d1", source_id="s")
        n2 = ConceptNode(id="d2:b", name="b", description="", domain="d2", source_id="s")
        backend.add_node(n1)
        backend.add_node(n2)
        backend.add_embedding("d1:a", [1.0, 0.0])
        backend.add_embedding("d2:b", [0.9, 0.1])

        results = backend.vector_search([1.0, 0.0], domain="d1", top_k=5)
        assert len(results) == 1
        assert results[0][0].id == "d1:a"

    def test_vector_search_with_index(self):
        from qortex.core.memory import InMemoryBackend
        from qortex.core.models import ConceptNode
        from qortex.vec.index import NumpyVectorIndex

        index = NumpyVectorIndex(dimensions=3)
        backend = InMemoryBackend(vector_index=index)
        backend.connect()
        backend.create_domain("test")

        node = ConceptNode(
            id="test:bar", name="bar", description="A bar concept",
            domain="test", source_id="s1",
        )
        backend.add_node(node)
        backend.add_embedding("test:bar", [0.0, 1.0, 0.0])

        # Index should have been populated
        assert index.size() == 1

        results = backend.vector_search([0.0, 1.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0][0].id == "test:bar"

    def test_supports_vector_search(self):
        from qortex.core.memory import InMemoryBackend

        backend = InMemoryBackend()
        assert not backend.supports_vector_search()  # No embeddings yet

        backend.add_embedding("x", [1.0])
        assert backend.supports_vector_search()

    def test_get_embedding(self):
        from qortex.core.memory import InMemoryBackend

        backend = InMemoryBackend()
        assert backend.get_embedding("x") is None
        backend.add_embedding("x", [1.0, 2.0])
        assert backend.get_embedding("x") == [1.0, 2.0]


# =============================================================================
# VecOnlyAdapter
# =============================================================================


class FakeEmbedding:
    """Fake embedding model for testing."""

    def __init__(self, dims=3):
        self._dims = dims

    def embed(self, texts):
        # Return a deterministic embedding based on text length
        return [[float(len(t) % 10) / 10, 0.5, 0.5] for t in texts]

    @property
    def dimensions(self):
        return self._dims


class TestVecOnlyAdapter:
    """Tests for VecOnlyAdapter retrieval."""

    def test_retrieve_returns_results(self):
        from qortex.core.memory import InMemoryBackend
        from qortex.core.models import ConceptNode
        from qortex.hippocampus.adapter import VecOnlyAdapter

        backend = InMemoryBackend()
        backend.connect()
        backend.create_domain("test")

        node = ConceptNode(
            id="test:concept1", name="concept1", description="A test concept",
            domain="test", source_id="s1",
        )
        backend.add_node(node)
        backend.add_embedding("test:concept1", [0.5, 0.5, 0.5])

        adapter = VecOnlyAdapter(backend, FakeEmbedding())
        result = adapter.retrieve("test query", top_k=5)

        assert len(result.items) > 0
        assert result.query_id  # Should have a query_id
        assert result.items[0].domain == "test"

    def test_retrieve_with_domain_filter(self):
        from qortex.core.memory import InMemoryBackend
        from qortex.core.models import ConceptNode
        from qortex.hippocampus.adapter import VecOnlyAdapter

        backend = InMemoryBackend()
        backend.connect()
        backend.create_domain("d1")
        backend.create_domain("d2")

        n1 = ConceptNode(id="d1:a", name="a", description="", domain="d1", source_id="s")
        n2 = ConceptNode(id="d2:b", name="b", description="", domain="d2", source_id="s")
        backend.add_node(n1)
        backend.add_node(n2)
        backend.add_embedding("d1:a", [0.5, 0.5, 0.5])
        backend.add_embedding("d2:b", [0.5, 0.5, 0.5])

        adapter = VecOnlyAdapter(backend, FakeEmbedding())
        result = adapter.retrieve("test", domains=["d1"], top_k=5)

        assert all(item.domain == "d1" for item in result.items)

    def test_feedback_is_noop(self):
        from qortex.core.memory import InMemoryBackend
        from qortex.hippocampus.adapter import VecOnlyAdapter

        backend = InMemoryBackend()
        adapter = VecOnlyAdapter(backend, FakeEmbedding())
        adapter.feedback("q1", {"item1": "accepted"})  # Should not raise
