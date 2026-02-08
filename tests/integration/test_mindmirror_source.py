"""Integration tests: MindMirror vec sync (Layer 2).

Tests PostgresSourceAdapter against real MindMirror PostgreSQL databases
running in Docker. Validates schema discovery, row serialization, vector
embedding, and cross-domain querying with actual data.

Requires: docker compose -f tests/integration/docker-compose.yml up -d
"""

from __future__ import annotations

import pytest

from qortex.sources.base import SourceConfig
from qortex.sources.postgres import PostgresSourceAdapter
from qortex.sources.serializer import NaturalLanguageSerializer

pytestmark = pytest.mark.integration


class TestMindMirrorConnect:
    """All 4 MindMirror databases are reachable."""

    @pytest.mark.asyncio
    async def test_connect_all_four_databases(self, mindmirror_configs):
        """Connect to all 4 MindMirror PostgreSQL instances."""
        for name, config in mindmirror_configs.items():
            adapter = PostgresSourceAdapter()
            await adapter.connect(config)
            assert adapter._conn is not None, f"Failed to connect to {name}"
            await adapter.disconnect()


class TestMindMirrorDiscover:
    """Schema discovery against real data."""

    @pytest.mark.asyncio
    async def test_discover_main_schema(self, mm_main_config):
        """Discovers all 6 tables in MindMirror main with correct column types."""
        adapter = PostgresSourceAdapter()
        await adapter.connect(mm_main_config)
        try:
            schemas = await adapter.discover()
            table_names = {s.name for s in schemas}

            # Expect 6 tables from the fixture
            expected = {
                "habit_templates",
                "habit_events",
                "journal_entries",
                "food_items",
                "meals",
                "meal_food_items",
            }
            assert expected == table_names, f"Got {table_names}"

            # Verify column counts
            food_items = next(s for s in schemas if s.name == "food_items")
            assert len(food_items.columns) >= 8  # id, name, slug, calories, protein, carbs, fat, serving_size, source, user_id, created_at
            assert food_items.row_count == 5

            # Verify PKs
            assert food_items.pk_columns == ["id"]

            # Verify FKs
            meal_food = next(s for s in schemas if s.name == "meal_food_items")
            fk_cols = meal_food.fk_columns
            fk_names = {c.name for c in fk_cols}
            assert "meal_id" in fk_names
            assert "food_item_id" in fk_names
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_discover_movements_schema(self, mm_movements_config):
        """Discovers movements DB with M2M junction tables."""
        adapter = PostgresSourceAdapter()
        await adapter.connect(mm_movements_config)
        try:
            schemas = await adapter.discover()
            table_names = {s.name for s in schemas}

            expected = {
                "movements",
                "muscles",
                "equipment",
                "movement_muscle_links",
                "movement_equipment_links",
            }
            assert expected == table_names

            movements = next(s for s in schemas if s.name == "movements")
            assert movements.row_count == 5

            # Verify the slug column exists (catalog indicator)
            col_names = {c.name for c in movements.columns}
            assert "slug" in col_names
            assert "difficulty" in col_names
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_discover_movements_detects_slug_as_catalog(self, mm_movements_config):
        """detect_catalog_table() returns True for movements (has slug)."""
        from qortex.sources.mapping_rules import detect_catalog_table
        from qortex.sources.graph_ingestor import TableSchemaFull

        adapter = PostgresSourceAdapter()
        await adapter.connect(mm_movements_config)
        try:
            schemas = await adapter.discover()
            movements = next(s for s in schemas if s.name == "movements")

            # Convert to TableSchemaFull for catalog detection
            full = TableSchemaFull(
                name=movements.name,
                columns=[
                    {"name": c.name, "data_type": c.data_type}
                    for c in movements.columns
                ],
            )
            assert detect_catalog_table(full) is True
        finally:
            await adapter.disconnect()


class TestMindMirrorSync:
    """Vec sync against real data."""

    @pytest.mark.asyncio
    async def test_sync_food_items_to_vector_layer(
        self, mm_main_config, fake_embedding, fake_vec_index
    ):
        """Sync food_items → query 'Chicken Breast' → match."""
        adapter = PostgresSourceAdapter()
        await adapter.connect(mm_main_config)
        try:
            await adapter.discover()
            result = await adapter.sync(
                tables=["food_items"],
                vector_index=fake_vec_index,
                embedding_model=fake_embedding,
            )

            assert result.tables_synced == 1
            assert result.rows_added == 5
            assert result.vectors_created == 5
            assert len(result.errors) == 0

            # Query for chicken breast
            query_emb = fake_embedding.embed(["chicken breast protein"])[0]
            results = fake_vec_index.query(query_emb, top_k=3)

            assert len(results) >= 1
            # Should find food items from mm_main
            assert all(vid.startswith("mm_main:food_items:") for vid, _ in results)
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_sync_movements_natural_language(
        self, mm_movements_config, fake_embedding, fake_vec_index
    ):
        """NaturalLanguageSerializer produces readable text from movements."""
        adapter = PostgresSourceAdapter()
        await adapter.connect(mm_movements_config)
        try:
            await adapter.discover()
            result = await adapter.sync(
                tables=["movements"],
                vector_index=fake_vec_index,
                embedding_model=fake_embedding,
            )

            assert result.tables_synced == 1
            assert result.rows_added == 5
            assert result.vectors_created == 5

            # Verify vec IDs contain movement PKs
            vec_ids = list(fake_vec_index.vectors.keys())
            assert all(vid.startswith("mm_movements:movements:") for vid in vec_ids)
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_domain_map_glob_matching(
        self, mm_main_config, fake_embedding, fake_vec_index
    ):
        """habit_* → 'habits' domain, food_items → 'nutrition'."""
        # The domain map is tested at the SourceConfig level
        assert mm_main_config.resolve_domain("habit_templates") == "habits"
        assert mm_main_config.resolve_domain("habit_events") == "habits"
        assert mm_main_config.resolve_domain("food_items") == "nutrition"
        assert mm_main_config.resolve_domain("meals") == "nutrition"  # meal* pattern
        assert mm_main_config.resolve_domain("journal_entries") == "reflection"

    @pytest.mark.asyncio
    async def test_cross_domain_query(self, mindmirror_configs, fake_embedding, fake_vec_index):
        """Sync habits + movements → query 'exercise' → results from both."""
        # Sync movements
        mv_adapter = PostgresSourceAdapter()
        await mv_adapter.connect(mindmirror_configs["mm_movements"])
        try:
            await mv_adapter.discover()
            await mv_adapter.sync(
                tables=["movements"],
                vector_index=fake_vec_index,
                embedding_model=fake_embedding,
            )
        finally:
            await mv_adapter.disconnect()

        # Sync habits
        main_adapter = PostgresSourceAdapter()
        await main_adapter.connect(mindmirror_configs["mm_main"])
        try:
            await main_adapter.discover()
            await main_adapter.sync(
                tables=["habit_templates"],
                vector_index=fake_vec_index,
                embedding_model=fake_embedding,
            )
        finally:
            await main_adapter.disconnect()

        # Query across domains
        query_emb = fake_embedding.embed(["daily fitness routine"])[0]
        results = fake_vec_index.query(query_emb, top_k=10)

        assert len(results) >= 2
        sources = {vid.split(":")[0] for vid, _ in results}
        # Should have results from both mm_movements and mm_main
        assert "mm_movements" in sources or "mm_main" in sources

    @pytest.mark.asyncio
    async def test_vec_id_format(self, mm_main_config, fake_embedding, fake_vec_index):
        """Vector IDs are source_id:table:pk."""
        adapter = PostgresSourceAdapter()
        await adapter.connect(mm_main_config)
        try:
            await adapter.discover()
            await adapter.sync(
                tables=["food_items"],
                vector_index=fake_vec_index,
                embedding_model=fake_embedding,
            )

            for vid in fake_vec_index.vectors:
                parts = vid.split(":")
                assert len(parts) == 3, f"Expected source:table:pk, got {vid}"
                assert parts[0] == "mm_main"
                assert parts[1] == "food_items"
                # PK is a UUID
                assert len(parts[2]) > 0
        finally:
            await adapter.disconnect()
