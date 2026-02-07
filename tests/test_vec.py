"""Comprehensive tests for the qortex vector layer.

Coverage:
- NumpyVectorIndex: unit, property, metamorphic
- SqliteVecIndex: skip-if-unavailable, parametrized parity with Numpy
- InMemoryBackend vector methods: unit, integration
- VecOnlyAdapter: unit, edge cases
- Ingestion embedding generation: integration
- EmbeddingModel protocol: conformance
"""

import math
import uuid

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, ExplicitRule, RelationType
from qortex.hippocampus.adapter import RetrievalItem, RetrievalResult, VecOnlyAdapter
from qortex.vec.index import NumpyVectorIndex

# =============================================================================
# Helpers / Fixtures
# =============================================================================


class FakeEmbedding:
    """Deterministic embedding model for testing."""

    def __init__(self, dims=3):
        self._dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Deterministic: hash text to produce a fixed vector."""
        results = []
        for t in texts:
            h = hash(t) & 0xFFFFFFFF
            vec = [((h >> (i * 8)) & 0xFF) / 255.0 for i in range(self._dims)]
            # Normalize so cosine sim is meaningful
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            results.append([x / norm for x in vec])
        return results

    @property
    def dimensions(self) -> int:
        return self._dims


class ControlledEmbedding:
    """Embedding model that returns pre-set vectors for specific texts."""

    def __init__(self, mapping: dict[str, list[float]], dims: int = 3):
        self._mapping = mapping
        self._dims = dims
        self._default = [0.0] * dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._mapping.get(t, self._default) for t in texts]

    @property
    def dimensions(self) -> int:
        return self._dims


def make_node(domain: str, name: str, desc: str = "") -> ConceptNode:
    return ConceptNode(
        id=f"{domain}:{name}", name=name, description=desc or f"A {name} concept",
        domain=domain, source_id="test-source",
    )


def make_backend_with_nodes(nodes: list[ConceptNode], embeddings: dict[str, list[float]] | None = None):
    """Create an InMemoryBackend pre-populated with nodes and optional embeddings."""
    backend = InMemoryBackend()
    backend.connect()
    domains = {n.domain for n in nodes}
    for d in domains:
        backend.create_domain(d)
    for n in nodes:
        backend.add_node(n)
    if embeddings:
        for nid, emb in embeddings.items():
            backend.add_embedding(nid, emb)
    return backend


# =============================================================================
# NumpyVectorIndex — Unit Tests
# =============================================================================


class TestNumpyVectorIndex:
    """Core unit tests for NumpyVectorIndex."""

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
        assert "b" not in [r[0] for r in results]

    def test_remove_nonexistent_id_is_silent(self):
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["a"], [[1, 0, 0]])
        idx.remove(["b", "c"])  # Should not raise
        assert idx.size() == 1

    def test_remove_all_then_search(self):
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["a", "b"], [[1, 0, 0], [0, 1, 0]])
        idx.remove(["a", "b"])
        assert idx.size() == 0
        assert idx.search([1, 0, 0]) == []

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
        assert idx.search([1, 0, 0]) == []

    def test_zero_vector_search(self):
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["a"], [[1, 0, 0]])
        assert idx.search([0, 0, 0]) == []

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
        idx.add(["a"], [[1.0, 0.0]])
        results = idx.search([1.0, 1.0], top_k=1)
        expected = 1.0 / math.sqrt(2)
        assert results[0][1] == pytest.approx(expected, abs=0.001)

    def test_persist_is_noop(self):
        idx = NumpyVectorIndex(dimensions=3)
        idx.persist()

    def test_large_batch_add(self):
        """Adding many vectors at once should work."""
        idx = NumpyVectorIndex(dimensions=8)
        n = 500
        ids = [f"id_{i}" for i in range(n)]
        vecs = np.random.randn(n, 8).tolist()
        idx.add(ids, vecs)
        assert idx.size() == n
        results = idx.search(vecs[0], top_k=1)
        assert results[0][0] == "id_0"

    def test_search_after_interleaved_add_remove(self):
        """Add, remove, add more — index stays consistent."""
        idx = NumpyVectorIndex(dimensions=2)
        idx.add(["a", "b"], [[1, 0], [0, 1]])
        idx.remove(["a"])
        idx.add(["c"], [[0.7, 0.7]])
        assert idx.size() == 2
        results = idx.search([1, 0], top_k=3)
        ids = [r[0] for r in results]
        assert "a" not in ids
        assert "c" in ids


# =============================================================================
# NumpyVectorIndex — Property-Based Tests
# =============================================================================


# Strategy for generating normalized vectors
def normalized_vector(dims):
    """Generate a random normalized vector of given dimensions."""
    return st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
                    min_size=dims, max_size=dims)


class TestNumpyVectorIndexProperties:
    """Property-based tests for NumpyVectorIndex."""

    @given(st.lists(st.text(min_size=1, max_size=10, alphabet="abcdefghij"),
                    min_size=1, max_size=20, unique=True))
    @settings(max_examples=30)
    def test_size_equals_unique_id_count(self, ids):
        """Size should always equal the number of unique IDs added."""
        dims = 3
        idx = NumpyVectorIndex(dimensions=dims)
        vecs = [[float(i + j) for j in range(dims)] for i in range(len(ids))]
        idx.add(ids, vecs)
        assert idx.size() == len(ids)

    @given(st.lists(st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
                    min_size=3, max_size=3))
    @settings(max_examples=30)
    def test_self_similarity_is_one(self, vec):
        """A vector's cosine similarity with itself should be 1.0."""
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["self"], [vec])
        results = idx.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    @given(st.integers(min_value=1, max_value=50))
    @settings(max_examples=15)
    def test_search_returns_at_most_top_k(self, k):
        """Search should never return more than top_k results."""
        idx = NumpyVectorIndex(dimensions=3)
        n = 100
        ids = [f"id_{i}" for i in range(n)]
        vecs = [[float(i), float(i + 1), float(i + 2)] for i in range(n)]
        idx.add(ids, vecs)
        results = idx.search([1, 2, 3], top_k=k)
        assert len(results) <= k

    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    @settings(max_examples=20)
    def test_all_scores_above_threshold(self, threshold):
        """All returned results should have score >= threshold."""
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["a", "b", "c"], [[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]])
        results = idx.search([1, 0, 0], top_k=10, threshold=threshold)
        for _, score in results:
            assert score >= threshold - 1e-6  # float tolerance


# =============================================================================
# NumpyVectorIndex — Metamorphic Tests
# =============================================================================


class TestNumpyVectorIndexMetamorphic:
    """Metamorphic relation tests: if we transform input, output transforms predictably."""

    def test_scaling_query_preserves_ranking(self):
        """Cosine sim is scale-invariant: 2*q should give same ranking as q."""
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["a", "b", "c"], [[1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]])
        r1 = idx.search([1, 0, 0], top_k=3)
        r2 = idx.search([10, 0, 0], top_k=3)
        assert [r[0] for r in r1] == [r[0] for r in r2]

    def test_scaling_query_preserves_scores(self):
        """Scores should be identical for scaled queries."""
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["a", "b"], [[1, 0, 0], [0.5, 0.5, 0]])
        r1 = idx.search([1, 0, 0], top_k=2)
        r2 = idx.search([100, 0, 0], top_k=2)
        for (_, s1), (_, s2) in zip(r1, r2):
            assert s1 == pytest.approx(s2, abs=0.001)

    def test_negating_query_reverses_preferences(self):
        """If q prefers a over b, -q should prefer b over a (when a and b are orthogonal)."""
        idx = NumpyVectorIndex(dimensions=2)
        idx.add(["pos", "neg"], [[1, 0], [-1, 0]])
        r_pos = idx.search([1, 0], top_k=2)
        r_neg = idx.search([-1, 0], top_k=2)
        assert r_pos[0][0] == "pos"
        assert r_neg[0][0] == "neg"

    def test_adding_irrelevant_vector_doesnt_change_top_result(self):
        """Adding an orthogonal vector shouldn't change the best match."""
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["a"], [[1, 0, 0]])
        best_before = idx.search([1, 0, 0], top_k=1)

        idx.add(["b"], [[0, 0, 1]])  # orthogonal
        best_after = idx.search([1, 0, 0], top_k=1)

        assert best_before[0][0] == best_after[0][0]
        assert best_before[0][1] == pytest.approx(best_after[0][1], abs=0.001)

    def test_remove_then_readd_gives_same_results(self):
        """Removing and re-adding a vector should give identical search results."""
        idx = NumpyVectorIndex(dimensions=3)
        idx.add(["a", "b"], [[1, 0, 0], [0, 1, 0]])
        r_before = idx.search([0.5, 0.5, 0], top_k=2)

        idx.remove(["a"])
        idx.add(["a"], [[1, 0, 0]])
        r_after = idx.search([0.5, 0.5, 0], top_k=2)

        assert set(r[0] for r in r_before) == set(r[0] for r in r_after)


# =============================================================================
# SqliteVecIndex — Skip-if-unavailable, parity tests
# =============================================================================

try:
    import sqlite_vec
    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

skip_no_sqlite_vec = pytest.mark.skipif(not HAS_SQLITE_VEC, reason="sqlite-vec not installed")


@skip_no_sqlite_vec
class TestSqliteVecIndex:
    """Tests for SqliteVecIndex (persistent, production)."""

    def _make_index(self, tmp_path, dims=3):
        from qortex.vec.index import SqliteVecIndex
        return SqliteVecIndex(db_path=str(tmp_path / "test_vectors.db"), dimensions=dims)

    def test_add_and_size(self, tmp_path):
        idx = self._make_index(tmp_path)
        assert idx.size() == 0
        idx.add(["a", "b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        assert idx.size() == 2

    def test_search_returns_results(self, tmp_path):
        idx = self._make_index(tmp_path)
        idx.add(["a", "b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        results = idx.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) > 0
        assert results[0][0] == "a"

    def test_remove(self, tmp_path):
        idx = self._make_index(tmp_path)
        idx.add(["a", "b"], [[1, 0, 0], [0, 1, 0]])
        idx.remove(["a"])
        assert idx.size() == 1

    def test_upsert_semantics(self, tmp_path):
        idx = self._make_index(tmp_path)
        idx.add(["a"], [[1, 0, 0]])
        idx.add(["a"], [[0, 1, 0]])
        assert idx.size() == 1

    def test_persistence_across_instances(self, tmp_path):
        """Data should persist across SqliteVecIndex instances."""
        db_path = str(tmp_path / "persist_test.db")

        from qortex.vec.index import SqliteVecIndex

        idx1 = SqliteVecIndex(db_path=db_path, dimensions=3)
        idx1.add(["a"], [[1, 0, 0]])
        idx1.persist()
        idx1.close()

        idx2 = SqliteVecIndex(db_path=db_path, dimensions=3)
        assert idx2.size() == 1
        results = idx2.search([1, 0, 0], top_k=1)
        assert results[0][0] == "a"
        idx2.close()

    def test_dimension_mismatch_raises(self, tmp_path):
        idx = self._make_index(tmp_path)
        with pytest.raises(ValueError, match="dims"):
            idx.add(["a"], [[1, 0]])


# =============================================================================
# Parametrized parity: NumpyVectorIndex and SqliteVecIndex same behavior
# =============================================================================

def get_index_factories(tmp_path):
    """Return list of (name, factory) for parametrized testing."""
    factories = [("numpy", lambda dims: NumpyVectorIndex(dimensions=dims))]
    if HAS_SQLITE_VEC:
        from qortex.vec.index import SqliteVecIndex
        factories.append((
            "sqlite",
            lambda dims, p=tmp_path: SqliteVecIndex(
                db_path=str(p / f"parity_{uuid.uuid4().hex[:8]}.db"), dimensions=dims
            ),
        ))
    return factories


class TestVectorIndexParity:
    """Both impls should return the same top-1 result for the same data."""

    def test_same_top_result(self, tmp_path):
        vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.5, 0.5, 0.0]]
        ids = ["a", "b", "c"]
        query = [0.8, 0.2, 0.0]

        results_by_impl = {}
        for name, factory in get_index_factories(tmp_path):
            idx = factory(3)
            idx.add(ids, vecs)
            results = idx.search(query, top_k=1)
            results_by_impl[name] = results[0][0]

        # All impls should agree on top result
        values = list(results_by_impl.values())
        assert all(v == values[0] for v in values), f"Disagreement: {results_by_impl}"


# =============================================================================
# InMemoryBackend vector integration
# =============================================================================


class TestInMemoryBackendVectorSearch:
    """Tests for vector search via InMemoryBackend."""

    def test_add_and_search_embedding(self):
        backend = InMemoryBackend()
        backend.connect()
        backend.create_domain("test")

        node = make_node("test", "foo")
        backend.add_node(node)
        backend.add_embedding("test:foo", [1.0, 0.0, 0.0])

        results = backend.vector_search([1.0, 0.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0][0].id == "test:foo"
        assert results[0][1] == pytest.approx(1.0, abs=0.01)

    def test_vector_search_with_domain_filter(self):
        backend = InMemoryBackend()
        backend.connect()
        backend.create_domain("d1")
        backend.create_domain("d2")

        backend.add_node(make_node("d1", "a"))
        backend.add_node(make_node("d2", "b"))
        backend.add_embedding("d1:a", [1.0, 0.0])
        backend.add_embedding("d2:b", [0.9, 0.1])

        results = backend.vector_search([1.0, 0.0], domain="d1", top_k=5)
        assert len(results) == 1
        assert results[0][0].id == "d1:a"

    def test_vector_search_with_index(self):
        index = NumpyVectorIndex(dimensions=3)
        backend = InMemoryBackend(vector_index=index)
        backend.connect()
        backend.create_domain("test")

        backend.add_node(make_node("test", "bar"))
        backend.add_embedding("test:bar", [0.0, 1.0, 0.0])

        assert index.size() == 1

        results = backend.vector_search([0.0, 1.0, 0.0], top_k=5)
        assert len(results) == 1
        assert results[0][0].id == "test:bar"

    def test_supports_vector_search(self):
        backend = InMemoryBackend()
        assert not backend.supports_vector_search()
        backend.add_embedding("x", [1.0])
        assert backend.supports_vector_search()

    def test_get_embedding(self):
        backend = InMemoryBackend()
        assert backend.get_embedding("x") is None
        backend.add_embedding("x", [1.0, 2.0])
        assert backend.get_embedding("x") == [1.0, 2.0]

    def test_vector_search_empty_backend(self):
        backend = InMemoryBackend()
        results = backend.vector_search([1.0, 0.0, 0.0])
        assert results == []

    def test_vector_search_respects_top_k(self):
        backend = InMemoryBackend()
        backend.connect()
        backend.create_domain("test")
        for i in range(10):
            backend.add_node(make_node("test", f"n{i}"))
            backend.add_embedding(f"test:n{i}", [float(i) / 10, 1.0, 0.0])

        results = backend.vector_search([0.5, 1.0, 0.0], top_k=3)
        assert len(results) == 3

    def test_embedding_overwrite(self):
        """Re-adding an embedding should overwrite."""
        backend = InMemoryBackend()
        backend.connect()
        backend.create_domain("test")
        backend.add_node(make_node("test", "x"))

        backend.add_embedding("test:x", [1.0, 0.0])
        backend.add_embedding("test:x", [0.0, 1.0])  # overwrite

        assert backend.get_embedding("test:x") == [0.0, 1.0]

    def test_vector_search_with_index_and_domain_filter(self):
        """Index-backed search should still respect domain filtering."""
        index = NumpyVectorIndex(dimensions=3)
        backend = InMemoryBackend(vector_index=index)
        backend.connect()
        backend.create_domain("d1")
        backend.create_domain("d2")

        backend.add_node(make_node("d1", "a"))
        backend.add_node(make_node("d2", "b"))
        backend.add_embedding("d1:a", [1.0, 0.0, 0.0])
        backend.add_embedding("d2:b", [0.95, 0.05, 0.0])

        results = backend.vector_search([1.0, 0.0, 0.0], domain="d2", top_k=5)
        assert len(results) == 1
        assert results[0][0].domain == "d2"

    def test_deleted_node_excluded_from_search(self):
        """If node is removed from backend, vector search shouldn't return it."""
        backend = InMemoryBackend()
        backend.connect()
        backend.create_domain("test")
        backend.add_node(make_node("test", "a"))
        backend.add_embedding("test:a", [1.0, 0.0])

        # Simulate node removal by deleting domain
        backend.delete_domain("test")

        # Embedding is orphaned — search should return nothing because node lookup fails
        results = backend.vector_search([1.0, 0.0], domain="test", top_k=5)
        assert len(results) == 0


# =============================================================================
# VecOnlyAdapter
# =============================================================================


class TestVecOnlyAdapter:
    """Tests for VecOnlyAdapter retrieval."""

    def test_retrieve_returns_results(self):
        backend = make_backend_with_nodes(
            [make_node("test", "concept1")],
            {"test:concept1": [0.5, 0.5, 0.5]},
        )
        adapter = VecOnlyAdapter(backend, FakeEmbedding())
        result = adapter.retrieve("test query", top_k=5)

        assert len(result.items) > 0
        assert result.query_id
        assert result.items[0].domain == "test"

    def test_retrieve_with_domain_filter(self):
        backend = make_backend_with_nodes(
            [make_node("d1", "a"), make_node("d2", "b")],
            {"d1:a": [0.5, 0.5, 0.5], "d2:b": [0.5, 0.5, 0.5]},
        )
        adapter = VecOnlyAdapter(backend, FakeEmbedding())
        result = adapter.retrieve("test", domains=["d1"], top_k=5)

        assert all(item.domain == "d1" for item in result.items)

    def test_retrieve_empty_backend(self):
        backend = InMemoryBackend()
        backend.connect()
        adapter = VecOnlyAdapter(backend, FakeEmbedding())
        result = adapter.retrieve("anything")

        assert len(result.items) == 0
        assert result.query_id  # Still has a query_id

    def test_retrieve_populates_all_fields(self):
        backend = make_backend_with_nodes(
            [make_node("test", "foo", "A foo thing")],
            {"test:foo": [0.5, 0.5, 0.5]},
        )
        adapter = VecOnlyAdapter(backend, FakeEmbedding())
        result = adapter.retrieve("test query")

        assert len(result.items) >= 1
        item = result.items[0]
        assert item.id == "test:foo"
        assert "foo" in item.content
        assert item.score > 0
        assert item.domain == "test"
        assert item.node_id == "test:foo"

    def test_retrieve_activated_nodes_populated(self):
        backend = make_backend_with_nodes(
            [make_node("test", "a"), make_node("test", "b")],
            {"test:a": [0.5, 0.5, 0.5], "test:b": [0.4, 0.4, 0.4]},
        )
        adapter = VecOnlyAdapter(backend, FakeEmbedding())
        result = adapter.retrieve("test", top_k=5)

        assert len(result.activated_nodes) == len(result.items)

    def test_retrieve_query_id_unique(self):
        backend = make_backend_with_nodes(
            [make_node("test", "x")],
            {"test:x": [1, 0, 0]},
        )
        adapter = VecOnlyAdapter(backend, FakeEmbedding())
        r1 = adapter.retrieve("q1")
        r2 = adapter.retrieve("q2")
        assert r1.query_id != r2.query_id

    def test_retrieve_with_multiple_domains(self):
        backend = make_backend_with_nodes(
            [make_node("d1", "a"), make_node("d2", "b"), make_node("d3", "c")],
            {"d1:a": [0.5, 0.5, 0.5], "d2:b": [0.5, 0.5, 0.5], "d3:c": [0.5, 0.5, 0.5]},
        )
        adapter = VecOnlyAdapter(backend, FakeEmbedding())
        result = adapter.retrieve("test", domains=["d1", "d3"], top_k=10)

        domains_returned = {item.domain for item in result.items}
        assert "d2" not in domains_returned

    def test_feedback_is_noop(self):
        backend = InMemoryBackend()
        adapter = VecOnlyAdapter(backend, FakeEmbedding())
        adapter.feedback("q1", {"item1": "accepted"})  # Should not raise

    def test_retrieve_respects_min_confidence(self):
        """High min_confidence should filter out low-scoring results."""
        backend = make_backend_with_nodes(
            [make_node("test", "close"), make_node("test", "far")],
            {"test:close": [1.0, 0.0, 0.0], "test:far": [0.0, 1.0, 0.0]},
        )
        # Use controlled embedding that produces a known query vector
        emb = ControlledEmbedding({"test query": [1.0, 0.0, 0.0]})
        adapter = VecOnlyAdapter(backend, emb)
        result = adapter.retrieve("test query", min_confidence=0.9, top_k=5)

        # Only "close" should survive the threshold
        assert len(result.items) == 1
        assert result.items[0].id == "test:close"


# =============================================================================
# Ingestion embedding integration
# =============================================================================


class TestIngestionEmbedding:
    """Integration tests: ingestor + embedding model → ConceptNodes with embeddings."""

    def test_ingest_with_embedding_populates_concept_embeddings(self):
        """Full pipeline: ingest text → concepts should have embeddings."""
        from qortex_ingest.base import Source, StubLLMBackend
        from qortex_ingest.text import TextIngestor

        stub_llm = StubLLMBackend(
            concepts=[
                {"name": "PatternA", "description": "First pattern"},
                {"name": "PatternB", "description": "Second pattern"},
            ],
            relations=[],
            rules=[],
        )

        embedding = FakeEmbedding(dims=3)
        ingestor = TextIngestor(llm=stub_llm, embedding_model=embedding)

        source = Source(raw_content="Some text about patterns.", source_type="text", name="test")
        manifest = ingestor.ingest(source, domain="test")

        assert len(manifest.concepts) == 2
        for concept in manifest.concepts:
            assert concept.embedding is not None, f"{concept.name} missing embedding"
            assert len(concept.embedding) == 3
            assert concept.embedding_model is not None

    def test_ingest_without_embedding_model_leaves_none(self):
        """Without embedding model, ConceptNodes should have embedding=None."""
        from qortex_ingest.base import Source, StubLLMBackend
        from qortex_ingest.text import TextIngestor

        stub_llm = StubLLMBackend(
            concepts=[{"name": "Concept1", "description": "Desc"}],
            relations=[],
            rules=[],
        )

        ingestor = TextIngestor(llm=stub_llm)
        source = Source(raw_content="Some text.", source_type="text", name="test")
        manifest = ingestor.ingest(source, domain="test")

        assert manifest.concepts[0].embedding is None

    def test_ingest_to_backend_roundtrip(self):
        """Ingest with embeddings → store in backend → vector_search returns results."""
        from qortex_ingest.base import Source, StubLLMBackend
        from qortex_ingest.text import TextIngestor

        stub_llm = StubLLMBackend(
            concepts=[
                {"name": "Alpha", "description": "The alpha concept"},
                {"name": "Beta", "description": "The beta concept"},
            ],
            relations=[],
            rules=[],
        )

        embedding = FakeEmbedding(dims=3)
        ingestor = TextIngestor(llm=stub_llm, embedding_model=embedding)

        source = Source(raw_content="Text about alpha and beta.", source_type="text", name="test")
        manifest = ingestor.ingest(source, domain="roundtrip")

        # Store in backend
        backend = InMemoryBackend()
        backend.connect()
        backend.ingest_manifest(manifest)

        # Also store embeddings
        for concept in manifest.concepts:
            if concept.embedding:
                backend.add_embedding(concept.id, concept.embedding)

        # Vector search should find concepts
        query_emb = embedding.embed(["Alpha: The alpha concept"])[0]
        results = backend.vector_search(query_emb, top_k=5)

        assert len(results) >= 1
        ids = [r[0].id for r in results]
        assert "roundtrip:Alpha" in ids

    def test_ingest_empty_concepts_with_embedding(self):
        """If LLM returns no concepts, embedding step is skipped gracefully."""
        from qortex_ingest.base import Source, StubLLMBackend
        from qortex_ingest.text import TextIngestor

        stub_llm = StubLLMBackend(concepts=[], relations=[], rules=[])
        embedding = FakeEmbedding(dims=3)
        ingestor = TextIngestor(llm=stub_llm, embedding_model=embedding)

        source = Source(raw_content="Empty.", source_type="text", name="test")
        manifest = ingestor.ingest(source, domain="empty")

        assert len(manifest.concepts) == 0  # No crash


# =============================================================================
# EmbeddingModel protocol conformance
# =============================================================================


class TestEmbeddingModelProtocol:
    """Verify FakeEmbedding and ControlledEmbedding satisfy the protocol."""

    def test_fake_embedding_implements_protocol(self):
        from qortex.vec.embeddings import EmbeddingModel
        emb = FakeEmbedding()
        assert isinstance(emb, EmbeddingModel)

    def test_controlled_embedding_implements_protocol(self):
        from qortex.vec.embeddings import EmbeddingModel
        emb = ControlledEmbedding({}, dims=3)
        assert isinstance(emb, EmbeddingModel)

    def test_fake_embedding_dimensions(self):
        assert FakeEmbedding(dims=5).dimensions == 5
        assert FakeEmbedding(dims=384).dimensions == 384

    def test_fake_embedding_batch_consistency(self):
        """Same text should always produce same embedding."""
        emb = FakeEmbedding(dims=3)
        r1 = emb.embed(["hello"])
        r2 = emb.embed(["hello"])
        assert r1 == r2

    def test_fake_embedding_output_shape(self):
        emb = FakeEmbedding(dims=5)
        results = emb.embed(["a", "b", "c"])
        assert len(results) == 3
        assert all(len(r) == 5 for r in results)


# =============================================================================
# RetrievalResult / RetrievalItem data model
# =============================================================================


class TestRetrievalDataModel:
    """Tests for the retrieval result data structures."""

    def test_retrieval_item_fields(self):
        item = RetrievalItem(id="x", content="hello", score=0.9, domain="d")
        assert item.id == "x"
        assert item.content == "hello"
        assert item.score == 0.9
        assert item.domain == "d"
        assert item.node_id is None
        assert item.metadata == {}

    def test_retrieval_result_defaults(self):
        result = RetrievalResult(items=[], query_id="q1")
        assert result.items == []
        assert result.query_id == "q1"
        assert result.activated_nodes == []

    def test_retrieval_result_with_items(self):
        items = [
            RetrievalItem(id="a", content="", score=0.9, domain="d"),
            RetrievalItem(id="b", content="", score=0.5, domain="d"),
        ]
        result = RetrievalResult(items=items, query_id="q2", activated_nodes=["a", "b"])
        assert len(result.items) == 2
        assert len(result.activated_nodes) == 2


# =============================================================================
# ConceptNode embedding fields
# =============================================================================


class TestConceptNodeEmbedding:
    """Tests for the embedding fields added to ConceptNode."""

    def test_default_embedding_is_none(self):
        node = ConceptNode(id="x", name="x", description="", domain="d", source_id="s")
        assert node.embedding is None
        assert node.embedding_model is None

    def test_embedding_can_be_set(self):
        node = ConceptNode(
            id="x", name="x", description="", domain="d", source_id="s",
            embedding=[1.0, 2.0, 3.0], embedding_model="test-model",
        )
        assert node.embedding == [1.0, 2.0, 3.0]
        assert node.embedding_model == "test-model"

    def test_embedding_preserved_through_backend(self):
        """Node embedding should survive add_node → get_node roundtrip."""
        backend = InMemoryBackend()
        backend.connect()
        backend.create_domain("test")

        node = ConceptNode(
            id="test:x", name="x", description="", domain="test", source_id="s",
            embedding=[1.0, 2.0], embedding_model="test",
        )
        backend.add_node(node)
        retrieved = backend.get_node("test:x")
        # InMemoryBackend stores the node object directly
        assert retrieved.embedding == [1.0, 2.0]


# =============================================================================
# get_adapter factory
# =============================================================================


class TestGetAdapter:
    """Tests for the adapter factory function."""

    def test_returns_vec_adapter_for_non_mage_backend(self):
        from qortex.hippocampus.adapter import VecOnlyAdapter, get_adapter

        backend = InMemoryBackend()  # supports_mage() = False
        adapter = get_adapter(backend, FakeEmbedding())
        assert isinstance(adapter, VecOnlyAdapter)

    def test_raises_without_embedding_model(self):
        from qortex.hippocampus.adapter import get_adapter

        backend = InMemoryBackend()
        with pytest.raises(ValueError, match="No embedding model"):
            get_adapter(backend, embedding_model=None)
