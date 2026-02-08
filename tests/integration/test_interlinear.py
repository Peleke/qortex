"""Integration tests: interlinear Supabase-like patterns.

Tests PostgresSourceAdapter against a real interlinear PostgreSQL database
running in Docker. Validates Supabase-like patterns: JSONB columns,
multilingual data, CHECK constraints, CASCADE deletes, M2M junctions.

Requires: docker compose -f tests/integration/docker-compose.yml up -d
"""

from __future__ import annotations

import pytest

from qortex.sources.postgres import PostgresSourceAdapter

pytestmark = pytest.mark.integration


class TestInterlinearDiscover:
    """Schema discovery for Supabase-like single DB."""

    @pytest.mark.asyncio
    async def test_discover_interlinear_schema(self, interlinear_config):
        """All 7 tables with JSONB, CHECK, CASCADE, M2M."""
        adapter = PostgresSourceAdapter()
        await adapter.connect(interlinear_config)
        try:
            schemas = await adapter.discover()
            table_names = {s.name for s in schemas}

            expected = {
                "courses",
                "lessons",
                "vocabulary",
                "lesson_vocabulary_items",
                "grammar_concepts",
                "exercises",
                "ai_generation_logs",
            }
            assert expected == table_names, f"Got {table_names}"

            # Verify courses has CHECK-constrained columns
            courses = next(s for s in schemas if s.name == "courses")
            col_names = {c.name for c in courses.columns}
            assert "language_code" in col_names
            assert "level" in col_names
            assert courses.row_count == 2

            # Verify exercises has JSONB column
            exercises = next(s for s in schemas if s.name == "exercises")
            options_col = next(
                (c for c in exercises.columns if c.name == "options"), None
            )
            assert options_col is not None
            assert "json" in options_col.data_type.lower()

            # Verify lesson_vocabulary_items is a junction table (M2M)
            lvi = next(s for s in schemas if s.name == "lesson_vocabulary_items")
            fk_names = {c.name for c in lvi.fk_columns}
            assert "lesson_id" in fk_names
            assert "vocabulary_id" in fk_names

            # Verify CASCADE FKs on lessons
            lessons = next(s for s in schemas if s.name == "lessons")
            fk_cols = lessons.fk_columns
            assert len(fk_cols) >= 1
            assert fk_cols[0].foreign_table == "courses"
        finally:
            await adapter.disconnect()


class TestInterlinearSync:
    """Vec sync for multilingual Supabase data."""

    @pytest.mark.asyncio
    async def test_sync_vocabulary_multilingual(
        self, interlinear_config, fake_embedding, fake_vec_index
    ):
        """Sync vocab → query 'hola' → match in es."""
        adapter = PostgresSourceAdapter()
        await adapter.connect(interlinear_config)
        try:
            await adapter.discover()
            result = await adapter.sync(
                tables=["vocabulary"],
                vector_index=fake_vec_index,
                embedding_model=fake_embedding,
            )

            assert result.tables_synced == 1
            assert result.rows_added == 5  # 5 vocab items (es + la)
            assert result.vectors_created == 5
            assert len(result.errors) == 0

            # Query for Spanish greeting
            query_emb = fake_embedding.embed(["hola hello greeting"])[0]
            results = fake_vec_index.query(query_emb, top_k=5)

            assert len(results) >= 1
            # All from interlinear:vocabulary
            assert all(
                vid.startswith("interlinear:vocabulary:") for vid, _ in results
            )
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_sync_courses_and_lessons(
        self, interlinear_config, fake_embedding, fake_vec_index
    ):
        """Sync both courses and lessons, verify both are indexed."""
        adapter = PostgresSourceAdapter()
        await adapter.connect(interlinear_config)
        try:
            await adapter.discover()
            result = await adapter.sync(
                tables=["courses", "lessons"],
                vector_index=fake_vec_index,
                embedding_model=fake_embedding,
            )

            assert result.tables_synced == 2
            assert result.rows_added >= 5  # 2 courses + 3 lessons
            assert len(result.errors) == 0

            # Verify both table sources exist
            vid_tables = {vid.split(":")[1] for vid in fake_vec_index.vectors}
            assert "courses" in vid_tables
            assert "lessons" in vid_tables
        finally:
            await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_ai_generation_metadata_domain(
        self, interlinear_config, fake_embedding, fake_vec_index
    ):
        """ai_generation_logs synced to 'metadata' domain."""
        assert interlinear_config.resolve_domain("ai_generation_logs") == "metadata"

    @pytest.mark.asyncio
    async def test_sync_all_tables(
        self, interlinear_config, fake_embedding, fake_vec_index
    ):
        """Sync all interlinear tables — no errors."""
        adapter = PostgresSourceAdapter()
        await adapter.connect(interlinear_config)
        try:
            await adapter.discover()
            result = await adapter.sync(
                vector_index=fake_vec_index,
                embedding_model=fake_embedding,
            )

            assert result.tables_synced == 7
            assert result.vectors_created >= 15  # At least 15 rows across all tables
            assert len(result.errors) == 0
        finally:
            await adapter.disconnect()
