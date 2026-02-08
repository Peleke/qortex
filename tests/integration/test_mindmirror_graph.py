"""Integration tests: MindMirror graph ingest (Layer 3).

Tests PostgresGraphIngestor against real MindMirror PostgreSQL databases
running in Docker. Validates schema discovery with constraints, FK→edge
classification, CHECK→rule extraction, catalog detection, and cross-DB
edge detection.

Requires: docker compose -f tests/integration/docker-compose.yml up -d
"""

from __future__ import annotations

import pytest

from qortex.core.memory import InMemoryBackend
from qortex.core.models import RelationType
from qortex.sources.cross_db import apply_cross_db_edges, discover_all_cross_db
from qortex.sources.postgres_graph import PostgresGraphIngestor

pytestmark = pytest.mark.integration


class TestGraphDiscover:
    """Schema discovery with constraint metadata."""

    @pytest.mark.asyncio
    async def test_discover_main_schema_with_constraints(self, mm_main_config):
        """Discover MindMirror main: FKs, CHECK constraints, row counts."""
        import asyncpg

        conn = await asyncpg.connect(mm_main_config.connection_string)
        try:
            ingestor = PostgresGraphIngestor(config=mm_main_config)
            schema = await ingestor.discover_schema(conn=conn)

            assert schema.source_id == "mm_main"
            table_names = {t.name for t in schema.tables}
            assert "habit_templates" in table_names
            assert "habit_events" in table_names
            assert "food_items" in table_names
            assert "meal_food_items" in table_names

            # Verify FK constraints on habit_events
            habit_events = schema.get_table("habit_events")
            assert habit_events is not None
            fk_cols = {fk.source_column for fk in habit_events.foreign_keys}
            assert "habit_template_id" in fk_cols

            # Verify CASCADE on habit_events → habit_templates
            ht_fk = next(
                fk for fk in habit_events.foreign_keys if fk.source_column == "habit_template_id"
            )
            assert ht_fk.on_delete == "CASCADE"

            # Verify CHECK constraints on food_items
            food_items = schema.get_table("food_items")
            assert food_items is not None
            check_names = [c.constraint_name for c in food_items.check_constraints]
            assert len(check_names) >= 1  # At least calories >= 0
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_discover_movements_check_constraints(self, mm_movements_config):
        """Discover movements DB: CHECK on difficulty, mechanics, laterality."""
        import asyncpg

        conn = await asyncpg.connect(mm_movements_config.connection_string)
        try:
            ingestor = PostgresGraphIngestor(config=mm_movements_config)
            schema = await ingestor.discover_schema(conn=conn)

            movements = schema.get_table("movements")
            assert movements is not None
            # difficulty IN (...), mechanics IN (...), laterality IN (...)
            assert len(movements.check_constraints) >= 3
        finally:
            await conn.close()


class TestGraphMapping:
    """map_schema() produces correct graph structure."""

    @pytest.mark.asyncio
    async def test_habit_cascade_part_of(self, mm_main_config):
        """habit_events.habit_template_id CASCADE → PART_OF edge."""
        import asyncpg

        conn = await asyncpg.connect(mm_main_config.connection_string)
        try:
            ingestor = PostgresGraphIngestor(config=mm_main_config)
            schema = await ingestor.discover_schema(conn=conn)
            mapping = ingestor.map_schema(schema, domain_map=mm_main_config.domain_map)

            # Find the edge mapping for habit_events → habit_templates
            edge = next(
                (
                    e
                    for e in mapping.edges
                    if e.source_table == "habit_events" and e.target_table == "habit_templates"
                ),
                None,
            )
            assert edge is not None
            # CASCADE → PART_OF (lifecycle coupling)
            assert edge.relation_type == RelationType.PART_OF.value
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_meal_food_junction_uses(self, mm_main_config):
        """meal_food_items (junction table) → USES edges."""
        import asyncpg

        conn = await asyncpg.connect(mm_main_config.connection_string)
        try:
            ingestor = PostgresGraphIngestor(config=mm_main_config)
            schema = await ingestor.discover_schema(conn=conn)
            mapping = ingestor.map_schema(schema, domain_map=mm_main_config.domain_map)

            # meal_food_items should produce USES edges (M2M junction)
            junction_edges = [e for e in mapping.edges if e.source_table == "meal_food_items"]
            assert len(junction_edges) == 2  # meal_id + food_item_id
            assert all(e.relation_type == RelationType.USES.value for e in junction_edges)
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_calories_check_constraint_rule(self, mm_main_config):
        """CHECK (calories >= 0) → ExplicitRule."""
        import asyncpg

        conn = await asyncpg.connect(mm_main_config.connection_string)
        try:
            ingestor = PostgresGraphIngestor(config=mm_main_config)
            schema = await ingestor.discover_schema(conn=conn)
            mapping = ingestor.map_schema(schema, domain_map=mm_main_config.domain_map)

            # Find rule about calories
            calorie_rules = [
                r
                for r in mapping.rules
                if r.table_name == "food_items" and "calories" in r.rule_text.lower()
            ]
            assert len(calorie_rules) >= 1
            assert "food_items" in calorie_rules[0].rule_text
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_catalog_detection_movements(self, mm_movements_config):
        """movements table detected as catalog (has slug column)."""
        import asyncpg

        conn = await asyncpg.connect(mm_movements_config.connection_string)
        try:
            ingestor = PostgresGraphIngestor(config=mm_movements_config)
            schema = await ingestor.discover_schema(conn=conn)
            mapping = ingestor.map_schema(schema, domain_map=mm_movements_config.domain_map)

            movements_tm = next((t for t in mapping.tables if t.table_name == "movements"), None)
            assert movements_tm is not None
            assert movements_tm.is_catalog is True
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_junction_not_catalog(self, mm_main_config):
        """meal_food_items (junction) should NOT be detected as catalog."""
        import asyncpg

        conn = await asyncpg.connect(mm_main_config.connection_string)
        try:
            ingestor = PostgresGraphIngestor(config=mm_main_config)
            schema = await ingestor.discover_schema(conn=conn)
            mapping = ingestor.map_schema(schema, domain_map=mm_main_config.domain_map)

            mfi_tm = next((t for t in mapping.tables if t.table_name == "meal_food_items"), None)
            assert mfi_tm is not None
            # Junction table with 2 FKs should NOT be catalog
            assert mfi_tm.is_catalog is False
        finally:
            await conn.close()


class TestGraphIngest:
    """Full graph ingest into InMemoryBackend."""

    @pytest.mark.asyncio
    async def test_habit_template_as_catalog_concept(self, mm_main_config, fake_embedding):
        """habit_templates rows → ConceptNode with name 'Morning Meditation'."""
        import asyncpg

        backend = InMemoryBackend()
        backend.connect()

        conn = await asyncpg.connect(mm_main_config.connection_string)
        try:
            ingestor = PostgresGraphIngestor(
                config=mm_main_config,
                backend=backend,
                embedding_model=fake_embedding,
            )
            schema = await ingestor.discover_schema(conn=conn)
            mapping = ingestor.map_schema(schema, domain_map=mm_main_config.domain_map)
            counts = await ingestor.ingest(mapping, schema, conn=conn)

            assert counts["concepts"] > 0
            assert counts["edges"] > 0

            # Find Morning Meditation concept
            meditation = None
            for node in backend._nodes.values():
                if "Morning Meditation" in node.name:
                    meditation = node
                    break
            assert meditation is not None
            assert meditation.domain == "habits"
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_full_main_db_ingest_counts(self, mm_main_config, fake_embedding):
        """Full ingest of mm_main: concepts, edges, rules all populated."""
        import asyncpg

        backend = InMemoryBackend()
        backend.connect()

        conn = await asyncpg.connect(mm_main_config.connection_string)
        try:
            ingestor = PostgresGraphIngestor(
                config=mm_main_config,
                backend=backend,
                embedding_model=fake_embedding,
            )
            schema = await ingestor.discover_schema(conn=conn)
            mapping = ingestor.map_schema(schema, domain_map=mm_main_config.domain_map)
            counts = await ingestor.ingest(mapping, schema, conn=conn)

            # 3 habit_templates + 3 habit_events + 2 journal_entries
            # + 5 food_items + 2 meals + 4 meal_food_items = 19
            assert counts["concepts"] >= 15
            # Edges from FKs: habit_events(2 FKs) * 3 rows + meal_food_items(2 FKs) * 4 + ...
            assert counts["edges"] >= 5
            # Rules from CHECK constraints: calories, protein, carbs, fat, meal_type, servings
            assert counts["rules"] >= 4
        finally:
            await conn.close()


class TestCrossDB:
    """Cross-database edge detection across MindMirror databases."""

    @pytest.mark.asyncio
    async def test_cross_db_movement_id_detection(self, mm_movements_config, mm_practices_config):
        """movement_templates.movement_id → movements in another DB."""
        import asyncpg

        conn_mv = await asyncpg.connect(mm_movements_config.connection_string)
        conn_pr = await asyncpg.connect(mm_practices_config.connection_string)
        try:
            # Discover both schemas
            mv_ingestor = PostgresGraphIngestor(config=mm_movements_config)
            mv_schema = await mv_ingestor.discover_schema(conn=conn_mv)

            pr_ingestor = PostgresGraphIngestor(config=mm_practices_config)
            pr_schema = await pr_ingestor.discover_schema(conn=conn_pr)

            # Detect cross-DB edges
            cross_edges = discover_all_cross_db([mv_schema, pr_schema])

            # movement_templates.movement_id should be detected
            mv_edges = [
                e
                for e in cross_edges
                if e.source_table == "movement_templates" and e.target_table == "movements"
            ]
            assert len(mv_edges) >= 1
            assert mv_edges[0].relation_type == RelationType.USES.value
        finally:
            await conn_mv.close()
            await conn_pr.close()

    @pytest.mark.asyncio
    async def test_cross_db_user_identity_edges(
        self, mm_main_config, mm_practices_config, mm_users_config
    ):
        """user_id columns across DBs → BELONGS_TO edges to users table."""
        import asyncpg

        conn_main = await asyncpg.connect(mm_main_config.connection_string)
        conn_pr = await asyncpg.connect(mm_practices_config.connection_string)
        conn_users = await asyncpg.connect(mm_users_config.connection_string)
        try:
            main_ingestor = PostgresGraphIngestor(config=mm_main_config)
            main_schema = await main_ingestor.discover_schema(conn=conn_main)

            pr_ingestor = PostgresGraphIngestor(config=mm_practices_config)
            pr_schema = await pr_ingestor.discover_schema(conn=conn_pr)

            users_ingestor = PostgresGraphIngestor(config=mm_users_config)
            users_schema = await users_ingestor.discover_schema(conn=conn_users)

            cross_edges = discover_all_cross_db([main_schema, pr_schema, users_schema])

            # user_id in habit_events, journal_entries, meals, practice_instances
            # all should point to users table
            user_edges = [
                e
                for e in cross_edges
                if e.target_table == "users" and e.relation_type == RelationType.BELONGS_TO.value
            ]
            # At least from mm_main and mm_practices (not mm_users itself)
            source_dbs = {e.source_database for e in user_edges}
            assert len(source_dbs) >= 2
        finally:
            await conn_main.close()
            await conn_pr.close()
            await conn_users.close()

    @pytest.mark.asyncio
    async def test_apply_cross_db_edges_creates_schema_nodes(
        self, mm_movements_config, mm_practices_config
    ):
        """apply_cross_db_edges() creates schema nodes and edges in backend."""
        import asyncpg

        backend = InMemoryBackend()
        backend.connect()

        conn_mv = await asyncpg.connect(mm_movements_config.connection_string)
        conn_pr = await asyncpg.connect(mm_practices_config.connection_string)
        try:
            mv_ingestor = PostgresGraphIngestor(config=mm_movements_config)
            mv_schema = await mv_ingestor.discover_schema(conn=conn_mv)

            pr_ingestor = PostgresGraphIngestor(config=mm_practices_config)
            pr_schema = await pr_ingestor.discover_schema(conn=conn_pr)

            cross_edges = discover_all_cross_db([mv_schema, pr_schema])

            # Build dummy node_id_maps so the apply function takes the non-fallback path
            node_id_maps = {
                mv_schema.database_name: {
                    "movements": {"dummy_pk": "mv:movements:dummy_pk"},
                },
                pr_schema.database_name: {
                    "movement_templates": {"dummy_pk": "pr:movement_templates:dummy_pk"},
                },
            }

            created = apply_cross_db_edges(cross_edges, backend, node_id_maps)
            assert created >= 1

            # Verify schema nodes were created
            schema_edges = [e for e in backend._edges if e.properties.get("cross_db")]
            assert len(schema_edges) >= 1
        finally:
            await conn_mv.close()
            await conn_pr.close()
