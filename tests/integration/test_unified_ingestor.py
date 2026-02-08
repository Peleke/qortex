"""Integration tests: PostgresIngestor unified API.

Tests the ergonomic unified ingestor that combines vec + graph layers
in a single run() call. Validates IngestConfig targets, user filtering,
and batch processing against real PostgreSQL data.

Requires: docker compose -f tests/integration/docker-compose.yml up -d
"""

from __future__ import annotations

import pytest

from qortex.sources.base import IngestConfig
from qortex.sources.postgres import PostgresIngestor

pytestmark = pytest.mark.integration


class TestUnifiedIngestor:
    """PostgresIngestor.run() integration tests."""

    @pytest.mark.asyncio
    async def test_from_url_vec_only(
        self, mm_movements_config, fake_embedding, fake_vec_index
    ):
        """IngestConfig(targets='vec') → only vec sync, no graph."""
        ingestor = PostgresIngestor(
            config=mm_movements_config,
            ingest=IngestConfig(targets="vec"),
            vector_index=fake_vec_index,
            embedding_model=fake_embedding,
        )

        result = await ingestor.run()

        assert result.source_id == "mm_movements"
        assert result.vec_result is not None
        assert result.vec_result.tables_synced >= 1
        assert result.vec_result.vectors_created >= 5  # 5 movements
        assert result.graph_result is None

    @pytest.mark.asyncio
    async def test_from_url_factory(self, fake_embedding, fake_vec_index):
        """from_url() factory creates working ingestor."""
        ingestor = PostgresIngestor.from_url(
            "postgresql://qortex_test:qortex_test@localhost:15435/swae_movements",
            source_id="mm_movements",
            domain_map={"*": "exercise"},
            ingest=IngestConfig(targets="vec"),
            vector_index=fake_vec_index,
            embedding_model=fake_embedding,
        )

        result = await ingestor.run()

        assert result.source_id == "mm_movements"
        assert result.vec_result is not None
        assert result.vec_result.vectors_created >= 5

    @pytest.mark.asyncio
    async def test_batch_mode(
        self, mm_main_config, fake_embedding, fake_vec_index
    ):
        """Batch processing respects batch_size."""
        ingestor = PostgresIngestor(
            config=mm_main_config,
            ingest=IngestConfig(targets="vec", batch_size=2),
            vector_index=fake_vec_index,
            embedding_model=fake_embedding,
        )

        result = await ingestor.run()

        assert result.vec_result is not None
        # All rows should still be synced even with small batches
        assert result.vec_result.vectors_created >= 10  # Multiple tables, multiple rows

    @pytest.mark.asyncio
    async def test_interlinear_full_sync(
        self, interlinear_config, fake_embedding, fake_vec_index
    ):
        """Interlinear Supabase-like DB: full vec sync."""
        ingestor = PostgresIngestor(
            config=interlinear_config,
            ingest=IngestConfig(targets="vec"),
            vector_index=fake_vec_index,
            embedding_model=fake_embedding,
        )

        result = await ingestor.run()

        assert result.vec_result is not None
        assert result.vec_result.tables_synced == 7
        assert result.vec_result.vectors_created >= 15
        assert len(result.vec_result.errors) == 0

    @pytest.mark.asyncio
    async def test_tables_discovered_count(
        self, mm_main_config, fake_embedding, fake_vec_index
    ):
        """tables_discovered reflects actual schema discovery."""
        ingestor = PostgresIngestor(
            config=mm_main_config,
            ingest=IngestConfig(targets="vec"),
            vector_index=fake_vec_index,
            embedding_model=fake_embedding,
        )

        result = await ingestor.run()

        assert result.tables_discovered == 6  # 6 tables in main
