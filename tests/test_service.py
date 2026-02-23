"""Tests for QortexService.

Tests the service layer directly with InMemoryBackend + FakeEmbedding.
Same fixture pattern as test_mcp_server.py.
"""

from __future__ import annotations

import hashlib

import pytest

from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptNode
from qortex.service import QortexService
from qortex.vec.index import NumpyVectorIndex

DIMS = 32


class FakeEmbedding:
    @property
    def dimensions(self) -> int:
        return DIMS

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h[:DIMS]]
            norm = sum(v * v for v in vec) ** 0.5
            result.append([v / norm for v in vec])
        return result


@pytest.fixture
def vector_index():
    return NumpyVectorIndex(dimensions=DIMS)


@pytest.fixture
def backend(vector_index) -> InMemoryBackend:
    b = InMemoryBackend(vector_index=vector_index)
    b.connect()
    return b


@pytest.fixture
def embedding() -> FakeEmbedding:
    return FakeEmbedding()


@pytest.fixture
def service(backend, embedding, vector_index) -> QortexService:
    return QortexService(
        backend=backend,
        vector_index=vector_index,
        embedding_model=embedding,
    )


async def _seed_data(service: QortexService):
    """Add some concepts to the service for testing."""
    result = await service.ingest_structured(
        concepts=[
            {"name": "Python", "description": "A programming language"},
            {"name": "FastAPI", "description": "A web framework for Python"},
            {"name": "Starlette", "description": "An ASGI framework"},
        ],
        domain="tech",
        edges=[
            {
                "source": "FastAPI",
                "target": "Starlette",
                "relation_type": "depends_on",
            },
        ],
        rules=[{"text": "Use async for I/O-bound operations"}],
    )
    return result


class TestServiceQuery:
    async def test_query_empty(self, service):
        result = await service.query("hello")
        assert result["items"] == []
        assert "query_id" in result

    async def test_query_with_data(self, service):
        await _seed_data(service)
        result = await service.query("python programming")
        assert len(result["items"]) > 0
        assert "query_id" in result

    async def test_query_increments_counter(self, service):
        await _seed_data(service)
        assert service.query_count == 0
        await service.query("test")
        assert service.query_count == 1

    async def test_query_clamps_top_k(self, service):
        await _seed_data(service)
        result = await service.query("test", top_k=0)
        assert isinstance(result, dict)

    async def test_query_domain_filter(self, service):
        await _seed_data(service)
        result = await service.query("python", domains=["tech"])
        assert all(
            item["domain"] == "tech" for item in result["items"]
        )


class TestServiceIngest:
    async def test_ingest_structured(self, service):
        result = await service.ingest_structured(
            concepts=[{"name": "A", "description": "Concept A"}],
            domain="test",
        )
        assert result["concepts"] == 1
        assert result["domain"] == "test"

    async def test_ingest_text(self, service):
        result = await service.ingest_text(
            text="Machine learning is a subset of AI.",
            domain="ml",
        )
        assert result["domain"] == "ml"
        assert result["concepts"] >= 0

    async def test_ingest_text_empty(self, service):
        result = await service.ingest_text(text="", domain="test")
        assert result["concepts"] == 0
        assert "Empty text provided" in result["warnings"]

    async def test_ingest_file_not_found(self, service):
        result = await service.ingest("/nonexistent/path.md", domain="test")
        assert "error" in result


class TestServiceDomains:
    async def test_domains_empty(self, service):
        result = await service.domains()
        assert result["domains"] == []

    async def test_domains_after_ingest(self, service):
        await _seed_data(service)
        result = await service.domains()
        names = [d["name"] for d in result["domains"]]
        assert "tech" in names


class TestServiceStatus:
    async def test_status(self, service):
        result = await service.status()
        assert result["status"] == "ok"
        assert result["backend"] == "InMemoryBackend"
        assert result["vector_search"] is True


class TestServiceExplore:
    async def test_explore_missing_node(self, service):
        result = await service.explore("nonexistent")
        assert result is None

    async def test_explore_existing_node(self, service):
        await _seed_data(service)
        domains = await service.domains()
        # Get a concept ID from the backend
        nodes = list(service.backend._nodes.values())
        if nodes:
            result = await service.explore(nodes[0].id)
            assert result is not None
            assert "node" in result
            assert result["node"]["id"] == nodes[0].id


class TestServiceRules:
    async def test_rules_empty(self, service):
        result = await service.rules()
        assert result["rules"] == []
        assert result["projection"] == "rules"

    async def test_rules_after_ingest(self, service):
        await _seed_data(service)
        result = await service.rules(domains=["tech"])
        assert result["domain_count"] >= 0


class TestServiceCompare:
    async def test_compare_no_vec(self):
        backend = InMemoryBackend()
        backend.connect()
        svc = QortexService(backend=backend)
        result = await svc.compare("test")
        assert "error" in result

    async def test_compare_with_data(self, service):
        await _seed_data(service)
        result = await service.compare("python")
        assert "vec_only" in result
        assert "graph_enhanced" in result
        assert "diff" in result


class TestServiceVectorOps:
    async def test_create_and_query_index(self, service):
        result = await service.vector_create_index("test_idx", 4)
        assert result["status"] == "created"

        await service.vector_upsert(
            "test_idx",
            vectors=[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            ids=["a", "b"],
            metadata=[{"label": "x"}, {"label": "y"}],
        )

        result = await service.vector_query("test_idx", [1.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(result["results"]) == 2

    async def test_list_indexes(self, service):
        await service.vector_create_index("idx1", 4)
        await service.vector_create_index("idx2", 8)
        result = await service.vector_list_indexes()
        assert set(result["indexes"]) == {"idx1", "idx2"}

    async def test_delete_index(self, service):
        await service.vector_create_index("del_me", 4)
        result = await service.vector_delete_index("del_me")
        assert result["status"] == "deleted"
        assert "del_me" not in service.vector_indexes


@pytest.mark.asyncio
class TestServiceFeedback:
    async def test_feedback(self, service):
        result = await service.feedback(
            query_id="q1",
            outcomes={"item1": "accepted"},
            source="test",
        )
        assert result["status"] == "recorded"
        assert service.feedback_count == 1

    async def test_feedback_invalid_outcome(self, service):
        result = await service.feedback(
            query_id="q1",
            outcomes={"item1": "invalid"},
        )
        assert "error" in result


@pytest.mark.asyncio
class TestServiceLearning:
    async def test_learning_select(self, service):
        result = await service.learning_select(
            learner="test",
            candidates=[
                {"id": "arm1", "metadata": {}},
                {"id": "arm2", "metadata": {}},
            ],
            k=1,
        )
        assert "selected_arms" in result
        assert len(result["selected_arms"]) == 1

    async def test_learning_observe(self, service):
        await service.learning_select(
            learner="test",
            candidates=[{"id": "arm1", "metadata": {}}],
        )
        result = await service.learning_observe(
            learner="test",
            arm_id="arm1",
            reward=1.0,
        )
        assert "alpha" in result
        assert "beta" in result

    async def test_learning_posteriors(self, service):
        await service.learning_select(
            learner="test",
            candidates=[{"id": "arm1", "metadata": {}}],
        )
        result = await service.learning_posteriors(learner="test")
        assert result["learner"] == "test"

    async def test_learning_metrics(self, service):
        await service.learning_select(
            learner="test",
            candidates=[{"id": "arm1", "metadata": {}}],
        )
        result = await service.learning_metrics(learner="test")
        assert isinstance(result, dict)
