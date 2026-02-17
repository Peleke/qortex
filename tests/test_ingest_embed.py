"""Tests for --embed flag on `qortex ingest load`."""

from __future__ import annotations

import hashlib
from unittest.mock import MagicMock

from qortex.cli.ingest import _embed_manifest_concepts
from qortex.core.models import ConceptNode, IngestionManifest, SourceMetadata
from qortex.vec.index import NumpyVectorIndex

DIMS = 32


class FakeEmbedding:
    """Deterministic hash-based embedding for testing."""

    @property
    def dimensions(self) -> int:
        return DIMS

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            h = hashlib.sha256(text.encode()).digest()
            vec = [float(b) / 255.0 for b in h[:DIMS]]
            result.append(vec)
        return result


def _make_manifest(n_concepts: int = 5, domain: str = "test_domain") -> IngestionManifest:
    """Create a manifest with N concepts that have descriptions."""
    source = SourceMetadata(
        id="test-source",
        name="Test Source",
        source_type="text",
        path_or_url="/dev/null",
    )
    concepts = [
        ConceptNode(
            id=f"{domain}:concept_{i}",
            name=f"Concept {i}",
            description=f"This is the description for concept {i} about software design.",
            domain=domain,
            source_id="test-source",
        )
        for i in range(n_concepts)
    ]
    return IngestionManifest(
        source=source,
        domain=domain,
        concepts=concepts,
        edges=[],
        rules=[],
    )


class TestEmbedManifestConcepts:
    """Tests for _embed_manifest_concepts helper."""

    def test_embeds_all_concepts_into_vec_index(self):
        """All concepts with descriptions get embedded and added to vec index."""
        manifest = _make_manifest(n_concepts=10)
        graph_backend = MagicMock()
        fake_model = FakeEmbedding()
        vec_index = NumpyVectorIndex(dimensions=DIMS)

        count = _embed_manifest_concepts(
            manifest, graph_backend,
            embedding_model=fake_model, vector_index=vec_index,
        )

        assert count == 10
        assert vec_index.size() == 10
        assert graph_backend.add_embedding.call_count == 10

    def test_skips_concepts_without_descriptions(self):
        """Concepts with empty descriptions are skipped."""
        manifest = _make_manifest(n_concepts=3)
        manifest.concepts[1] = ConceptNode(
            id="test_domain:no_desc",
            name="No Desc",
            description="",
            domain="test_domain",
            source_id="test-source",
        )
        graph_backend = MagicMock()
        fake_model = FakeEmbedding()
        vec_index = NumpyVectorIndex(dimensions=DIMS)

        count = _embed_manifest_concepts(
            manifest, graph_backend,
            embedding_model=fake_model, vector_index=vec_index,
        )

        assert count == 2
        assert vec_index.size() == 2
        assert graph_backend.add_embedding.call_count == 2

    def test_batching_works_with_large_manifest(self):
        """Manifests with >64 concepts are embedded in batches."""
        manifest = _make_manifest(n_concepts=150)
        graph_backend = MagicMock()
        fake_model = FakeEmbedding()
        vec_index = NumpyVectorIndex(dimensions=DIMS)

        count = _embed_manifest_concepts(
            manifest, graph_backend,
            embedding_model=fake_model, vector_index=vec_index,
        )

        assert count == 150
        assert vec_index.size() == 150
        assert graph_backend.add_embedding.call_count == 150

    def test_embedded_concepts_are_searchable(self):
        """Embedded concepts can be found via vec search."""
        manifest = _make_manifest(n_concepts=5)
        graph_backend = MagicMock()
        fake_model = FakeEmbedding()
        vec_index = NumpyVectorIndex(dimensions=DIMS)

        _embed_manifest_concepts(
            manifest, graph_backend,
            embedding_model=fake_model, vector_index=vec_index,
        )

        # Search with the same text as concept 0 — should be top hit
        query_emb = fake_model.embed(
            ["This is the description for concept 0 about software design."]
        )[0]
        results = vec_index.search(query_emb, top_k=3)

        assert len(results) > 0
        assert results[0][0] == "test_domain:concept_0"
        assert results[0][1] > 0.99  # near-perfect match

    def test_returns_zero_for_empty_manifest(self):
        """Returns 0 when manifest has no concepts."""
        manifest = _make_manifest(n_concepts=0)
        graph_backend = MagicMock()
        fake_model = FakeEmbedding()
        vec_index = NumpyVectorIndex(dimensions=DIMS)

        count = _embed_manifest_concepts(
            manifest, graph_backend,
            embedding_model=fake_model, vector_index=vec_index,
        )

        assert count == 0
        assert vec_index.size() == 0
