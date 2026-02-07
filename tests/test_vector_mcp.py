"""Tests for vector-level MCP tools (qortex_vector_* family).

These tools expose raw VectorIndex operations for MastraVector and
other vector-level consumers. Separate from text-level tools.
"""

import pytest

from qortex.core.memory import InMemoryBackend
from qortex.mcp.server import (
    _match_filter,
    _vector_create_index_impl,
    _vector_delete_impl,
    _vector_delete_index_impl,
    _vector_delete_many_impl,
    _vector_describe_index_impl,
    _vector_list_indexes_impl,
    _vector_query_impl,
    _vector_update_impl,
    _vector_upsert_impl,
    create_server,
)
from qortex.vec.index import NumpyVectorIndex


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeEmbedding:
    """Deterministic embedding for tests."""

    dimensions = 4

    def embed(self, texts):
        import hashlib

        results = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            vec = [float(b) / 255.0 for b in h[:4]]
            results.append(vec)
        return results


@pytest.fixture(autouse=True)
def _setup_server():
    """Create a fresh server for each test."""
    vec = NumpyVectorIndex(dimensions=4)
    backend = InMemoryBackend(vector_index=vec)
    backend.connect()
    create_server(backend=backend, embedding_model=FakeEmbedding(), vector_index=vec)
    yield


@pytest.fixture
def index_name():
    return "test_index"


@pytest.fixture
def created_index(index_name):
    """Create and return an index with 4 dimensions."""
    _vector_create_index_impl(index_name, dimension=4, metric="cosine")
    return index_name


@pytest.fixture
def populated_index(created_index):
    """An index with 3 vectors + metadata."""
    _vector_upsert_impl(
        created_index,
        vectors=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        metadata=[
            {"source": "doc1", "category": "auth"},
            {"source": "doc2", "category": "auth"},
            {"source": "doc3", "category": "billing"},
        ],
        ids=["v1", "v2", "v3"],
        documents=["OAuth2 framework", "JWT tokens", "Payment processing"],
    )
    return created_index


# ---------------------------------------------------------------------------
# Filter matching
# ---------------------------------------------------------------------------


class TestMatchFilter:
    def test_equality(self):
        assert _match_filter({"a": 1}, {"a": 1})
        assert not _match_filter({"a": 1}, {"a": 2})

    def test_missing_key(self):
        assert not _match_filter({}, {"a": 1})

    def test_ne(self):
        assert _match_filter({"a": 1}, {"a": {"$ne": 2}})
        assert not _match_filter({"a": 1}, {"a": {"$ne": 1}})

    def test_gt_gte_lt_lte(self):
        meta = {"score": 5}
        assert _match_filter(meta, {"score": {"$gt": 4}})
        assert not _match_filter(meta, {"score": {"$gt": 5}})
        assert _match_filter(meta, {"score": {"$gte": 5}})
        assert _match_filter(meta, {"score": {"$lt": 6}})
        assert not _match_filter(meta, {"score": {"$lt": 5}})
        assert _match_filter(meta, {"score": {"$lte": 5}})

    def test_in_nin(self):
        meta = {"cat": "auth"}
        assert _match_filter(meta, {"cat": {"$in": ["auth", "billing"]}})
        assert not _match_filter(meta, {"cat": {"$in": ["billing"]}})
        assert _match_filter(meta, {"cat": {"$nin": ["billing"]}})
        assert not _match_filter(meta, {"cat": {"$nin": ["auth"]}})

    def test_and(self):
        meta = {"a": 1, "b": 2}
        assert _match_filter(meta, {"$and": [{"a": 1}, {"b": 2}]})
        assert not _match_filter(meta, {"$and": [{"a": 1}, {"b": 3}]})

    def test_or(self):
        meta = {"a": 1}
        assert _match_filter(meta, {"$or": [{"a": 1}, {"a": 2}]})
        assert not _match_filter(meta, {"$or": [{"a": 2}, {"a": 3}]})

    def test_not(self):
        assert _match_filter({"a": 1}, {"$not": {"a": 2}})
        assert not _match_filter({"a": 1}, {"$not": {"a": 1}})

    def test_nested_operators(self):
        meta = {"source": "doc1", "score": 8}
        filt = {
            "$and": [
                {"source": "doc1"},
                {"score": {"$gte": 5}},
            ]
        }
        assert _match_filter(meta, filt)


# ---------------------------------------------------------------------------
# Index lifecycle
# ---------------------------------------------------------------------------


class TestCreateIndex:
    def test_create(self, index_name):
        result = _vector_create_index_impl(index_name, 4, "cosine")
        assert result["status"] == "created"

    def test_create_duplicate_same_dim(self, created_index):
        result = _vector_create_index_impl(created_index, 4, "cosine")
        assert result["status"] == "exists"

    def test_create_duplicate_different_dim(self, created_index):
        result = _vector_create_index_impl(created_index, 8, "cosine")
        assert "error" in result


class TestListIndexes:
    def test_empty(self):
        result = _vector_list_indexes_impl()
        assert result["indexes"] == []

    def test_with_indexes(self):
        _vector_create_index_impl("idx_a", 4)
        _vector_create_index_impl("idx_b", 8)
        result = _vector_list_indexes_impl()
        assert sorted(result["indexes"]) == ["idx_a", "idx_b"]


class TestDescribeIndex:
    def test_describe(self, created_index):
        result = _vector_describe_index_impl(created_index)
        assert result["dimension"] == 4
        assert result["count"] == 0
        assert result["metric"] == "cosine"

    def test_describe_after_upsert(self, populated_index):
        result = _vector_describe_index_impl(populated_index)
        assert result["count"] == 3

    def test_describe_missing(self):
        result = _vector_describe_index_impl("nope")
        assert "error" in result


class TestDeleteIndex:
    def test_delete(self, created_index):
        result = _vector_delete_index_impl(created_index)
        assert result["status"] == "deleted"
        assert _vector_list_indexes_impl()["indexes"] == []

    def test_delete_missing(self):
        result = _vector_delete_index_impl("nope")
        assert "error" in result


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------


class TestUpsert:
    def test_upsert_with_ids(self, created_index):
        result = _vector_upsert_impl(
            created_index,
            vectors=[[1, 0, 0, 0], [0, 1, 0, 0]],
            ids=["a", "b"],
        )
        assert result["ids"] == ["a", "b"]
        assert _vector_describe_index_impl(created_index)["count"] == 2

    def test_upsert_auto_ids(self, created_index):
        result = _vector_upsert_impl(
            created_index,
            vectors=[[1, 0, 0, 0]],
        )
        assert len(result["ids"]) == 1
        assert _vector_describe_index_impl(created_index)["count"] == 1

    def test_upsert_with_metadata(self, created_index):
        _vector_upsert_impl(
            created_index,
            vectors=[[1, 0, 0, 0]],
            metadata=[{"source": "test"}],
            ids=["m1"],
        )
        results = _vector_query_impl(created_index, [1, 0, 0, 0], top_k=1)
        assert results["results"][0]["metadata"]["source"] == "test"

    def test_upsert_with_documents(self, created_index):
        _vector_upsert_impl(
            created_index,
            vectors=[[1, 0, 0, 0]],
            documents=["Hello world"],
            ids=["d1"],
        )
        results = _vector_query_impl(created_index, [1, 0, 0, 0], top_k=1)
        assert results["results"][0]["document"] == "Hello world"

    def test_upsert_missing_index(self):
        result = _vector_upsert_impl("nope", vectors=[[1, 0, 0, 0]])
        assert "error" in result

    def test_upsert_overwrites(self, created_index):
        _vector_upsert_impl(created_index, vectors=[[1, 0, 0, 0]], ids=["x"], metadata=[{"v": 1}])
        _vector_upsert_impl(created_index, vectors=[[0, 1, 0, 0]], ids=["x"], metadata=[{"v": 2}])
        assert _vector_describe_index_impl(created_index)["count"] == 1
        results = _vector_query_impl(created_index, [0, 1, 0, 0], top_k=1)
        assert results["results"][0]["id"] == "x"
        assert results["results"][0]["metadata"]["v"] == 2


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_basic_query(self, populated_index):
        results = _vector_query_impl(populated_index, [1, 0, 0, 0], top_k=1)
        assert len(results["results"]) == 1
        assert results["results"][0]["id"] == "v1"
        assert results["results"][0]["score"] > 0

    def test_top_k(self, populated_index):
        results = _vector_query_impl(populated_index, [1, 0, 0, 0], top_k=2)
        assert len(results["results"]) == 2

    def test_with_filter(self, populated_index):
        results = _vector_query_impl(
            populated_index,
            [1, 0, 0, 0],
            top_k=10,
            filter={"category": "billing"},
        )
        assert len(results["results"]) == 1
        assert results["results"][0]["id"] == "v3"

    def test_filter_no_match(self, populated_index):
        results = _vector_query_impl(
            populated_index,
            [1, 0, 0, 0],
            top_k=10,
            filter={"category": "nonexistent"},
        )
        assert len(results["results"]) == 0

    def test_documents_in_results(self, populated_index):
        results = _vector_query_impl(populated_index, [1, 0, 0, 0], top_k=1)
        assert results["results"][0]["document"] == "OAuth2 framework"

    def test_missing_index(self):
        result = _vector_query_impl("nope", [1, 0, 0, 0])
        assert "error" in result


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


class TestUpdate:
    def test_update_metadata_by_id(self, populated_index):
        result = _vector_update_impl(
            populated_index, id="v1", metadata={"status": "archived"}
        )
        assert result["count"] == 1
        q = _vector_query_impl(populated_index, [1, 0, 0, 0], top_k=1)
        meta = q["results"][0]["metadata"]
        assert meta["status"] == "archived"
        assert meta["source"] == "doc1"  # Original preserved

    def test_update_vector_by_id(self, populated_index):
        _vector_update_impl(populated_index, id="v1", vector=[0, 0, 0, 1])
        q = _vector_query_impl(populated_index, [0, 0, 0, 1], top_k=1)
        assert q["results"][0]["id"] == "v1"

    def test_update_by_filter(self, populated_index):
        result = _vector_update_impl(
            populated_index,
            filter={"category": "auth"},
            metadata={"reviewed": True},
        )
        assert result["count"] == 2

    def test_update_no_target(self, populated_index):
        result = _vector_update_impl(populated_index)
        assert "error" in result

    def test_update_both_id_and_filter(self, populated_index):
        result = _vector_update_impl(
            populated_index, id="v1", filter={"a": 1}, metadata={"x": 1}
        )
        assert "error" in result

    def test_update_no_updates(self, populated_index):
        result = _vector_update_impl(populated_index, id="v1")
        assert "error" in result


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_single(self, populated_index):
        result = _vector_delete_impl(populated_index, "v1")
        assert result["status"] == "deleted"
        assert _vector_describe_index_impl(populated_index)["count"] == 2

    def test_delete_missing_index(self):
        result = _vector_delete_impl("nope", "v1")
        assert "error" in result


class TestDeleteMany:
    def test_delete_by_ids(self, populated_index):
        result = _vector_delete_many_impl(populated_index, ids=["v1", "v2"])
        assert result["count"] == 2
        assert _vector_describe_index_impl(populated_index)["count"] == 1

    def test_delete_by_filter(self, populated_index):
        result = _vector_delete_many_impl(
            populated_index, filter={"category": "auth"}
        )
        assert result["count"] == 2
        assert _vector_describe_index_impl(populated_index)["count"] == 1

    def test_delete_no_target(self, populated_index):
        result = _vector_delete_many_impl(populated_index)
        assert "error" in result

    def test_delete_both_ids_and_filter(self, populated_index):
        result = _vector_delete_many_impl(
            populated_index, ids=["v1"], filter={"a": 1}
        )
        assert "error" in result


# ---------------------------------------------------------------------------
# Full lifecycle (MastraVector simulation)
# ---------------------------------------------------------------------------


class TestMastraVectorLifecycle:
    """Simulates the full MastraVector workflow via MCP tools."""

    def test_full_lifecycle(self):
        # 1. Create index
        r = _vector_create_index_impl("docs", 4, "cosine")
        assert r["status"] == "created"

        # 2. Upsert vectors with metadata
        r = _vector_upsert_impl(
            "docs",
            vectors=[[1, 0, 0, 0], [0, 1, 0, 0], [0.7, 0.7, 0, 0]],
            metadata=[
                {"text": "OAuth2 auth", "source_id": "sec.pdf"},
                {"text": "JWT tokens", "source_id": "sec.pdf"},
                {"text": "API keys", "source_id": "api.pdf"},
            ],
            ids=["doc_1", "doc_2", "doc_3"],
        )
        assert len(r["ids"]) == 3

        # 3. Describe index
        r = _vector_describe_index_impl("docs")
        assert r["count"] == 3
        assert r["dimension"] == 4

        # 4. Query
        r = _vector_query_impl("docs", [1, 0, 0, 0], top_k=2)
        assert len(r["results"]) == 2
        assert r["results"][0]["id"] == "doc_1"  # Best match

        # 5. Query with filter
        r = _vector_query_impl(
            "docs", [1, 0, 0, 0], top_k=10,
            filter={"source_id": "api.pdf"},
        )
        assert len(r["results"]) == 1
        assert r["results"][0]["id"] == "doc_3"

        # 6. Update metadata
        _vector_update_impl("docs", id="doc_1", metadata={"reviewed": True})
        r = _vector_query_impl("docs", [1, 0, 0, 0], top_k=1)
        assert r["results"][0]["metadata"]["reviewed"] is True

        # 7. Delete single
        _vector_delete_impl("docs", "doc_2")
        assert _vector_describe_index_impl("docs")["count"] == 2

        # 8. Delete by filter
        _vector_delete_many_impl("docs", filter={"source_id": "sec.pdf"})
        assert _vector_describe_index_impl("docs")["count"] == 1

        # 9. List and describe
        r = _vector_list_indexes_impl()
        assert "docs" in r["indexes"]

        # 10. Delete index
        _vector_delete_index_impl("docs")
        assert _vector_list_indexes_impl()["indexes"] == []
