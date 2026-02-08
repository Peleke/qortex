"""Tests for PostgresGraphIngestor and cross-DB edge detection.

All tests mock asyncpg — no real database required.
Uses InMemoryBackend for real graph operations.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from qortex.core.memory import InMemoryBackend
from qortex.core.models import RelationType
from qortex.sources.graph_ingestor import (
    CheckConstraintInfo,
    CrossDatabaseEdge,
    EdgeMapping,
    ForeignKeyInfo,
    GraphMapping,
    RuleMapping,
    SchemaGraph,
    TableMapping,
    TableSchemaFull,
    UniqueConstraintInfo,
)
from qortex.sources.mapping_rules import (
    auto_domain,
    classify_fk_relation,
    constraint_to_rule,
    create_user_identity_edges,
    detect_catalog_table,
    detect_cross_db_edges_by_naming,
    find_description_columns,
    find_name_column,
)
from qortex.sources.postgres_graph import PostgresGraphIngestor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _col(name: str, data_type: str = "text", is_pk: bool = False, **kw) -> dict:
    """Build a column dict for TableSchemaFull."""
    d = {"name": name, "data_type": data_type, "nullable": True, "is_pk": is_pk}
    d.update(kw)
    return d


def _fk(
    source_table: str,
    source_column: str,
    target_table: str,
    target_column: str = "id",
    on_delete: str = "NO ACTION",
    constraint_name: str | None = None,
) -> ForeignKeyInfo:
    """Build a ForeignKeyInfo."""
    return ForeignKeyInfo(
        constraint_name=constraint_name or f"fk_{source_table}_{source_column}",
        source_table=source_table,
        source_column=source_column,
        target_table=target_table,
        target_column=target_column,
        on_delete=on_delete,
    )


def _make_schema_graph(
    source_id: str,
    database_name: str,
    tables: list[TableSchemaFull],
) -> SchemaGraph:
    return SchemaGraph(source_id=source_id, database_name=database_name, tables=tables)


def _make_mock_conn(tables_data: dict[str, list[dict]]) -> AsyncMock:
    """Create a mock asyncpg conn that returns rows from tables_data.

    tables_data: {table_name: [row_dicts]}
    """
    conn = AsyncMock()

    async def mock_fetch(query: str, *args):
        q = query.strip()

        if "information_schema.tables" in q:
            return [{"table_name": t} for t in tables_data]

        if "information_schema.columns" in q:
            table_name = args[1] if len(args) > 1 else None
            if table_name and table_name in tables_data:
                rows = tables_data[table_name]
                if rows:
                    return [
                        {
                            "column_name": col,
                            "data_type": "text",
                            "is_nullable": "YES",
                            "column_default": None,
                        }
                        for col in rows[0].keys()
                    ]
            return []

        if "table_constraints" in q and "PRIMARY KEY" in q:
            table_name = args[1] if len(args) > 1 else None
            # Assume first column is PK
            if table_name and table_name in tables_data:
                rows = tables_data[table_name]
                if rows:
                    first_col = list(rows[0].keys())[0]
                    return [{"column_name": first_col}]
            return []

        if "pg_constraint" in q and "contype = 'f'" in q:
            return []  # No FKs by default

        if "pg_constraint" in q and "contype = 'c'" in q:
            return []  # No CHECK by default

        if "pg_constraint" in q and "contype = 'u'" in q:
            return []  # No UNIQUE by default

        if "SELECT *" in q.upper():
            for tname, rows in tables_data.items():
                if tname in query:
                    return rows
            return []

        return []

    async def mock_fetchval(query: str, *args):
        if "COUNT" in query.upper():
            for tname, rows in tables_data.items():
                if tname in query:
                    return len(rows)
        return 0

    conn.fetch = mock_fetch
    conn.fetchval = mock_fetchval
    conn.close = AsyncMock()
    return conn


class FakeEmbedding:
    """Fake embedding model."""

    dimensions = 4

    def embed(self, texts: list[str]) -> list[list[float]]:
        import numpy as np
        return [np.random.default_rng(42).random(4).tolist() for _ in texts]


# ---------------------------------------------------------------------------
# TestPostgresGraphDiscover
# ---------------------------------------------------------------------------


class TestPostgresGraphDiscover:
    """Schema discovery with full constraint info."""

    @pytest.mark.asyncio
    async def test_basic_schema(self):
        """Discovers tables with columns and PKs."""
        conn = _make_mock_conn({
            "movements": [
                {"id": "mv1", "name": "Squat", "slug": "squat"},
                {"id": "mv2", "name": "Deadlift", "slug": "deadlift"},
            ],
        })

        ingestor = PostgresGraphIngestor()
        schema = await ingestor.discover_schema(
            conn, database_name="swae_movements", source_id="mm_movements"
        )

        assert schema.source_id == "mm_movements"
        assert schema.database_name == "swae_movements"
        assert len(schema.tables) == 1

        movements = schema.tables[0]
        assert movements.name == "movements"
        assert movements.row_count == 2

    @pytest.mark.asyncio
    async def test_multiple_tables(self):
        """Discovers multiple tables."""
        conn = _make_mock_conn({
            "users": [{"id": "u1", "email": "a@b.com"}],
            "roles": [{"id": "r1", "name": "admin"}],
        })

        ingestor = PostgresGraphIngestor()
        schema = await ingestor.discover_schema(conn, source_id="test")

        assert len(schema.tables) == 2
        table_names = {t.name for t in schema.tables}
        assert table_names == {"users", "roles"}

    @pytest.mark.asyncio
    async def test_requires_connection(self):
        """discover_schema raises without connection."""
        ingestor = PostgresGraphIngestor()
        with pytest.raises(RuntimeError, match="No connection"):
            await ingestor.discover_schema(None)


# ---------------------------------------------------------------------------
# TestGraphMapping
# ---------------------------------------------------------------------------


class TestGraphMapping:
    """Graph mapping from schema to GraphMapping."""

    def test_fk_edge_types(self):
        """FK → correct RelationType classification."""
        # user_id → BELONGS_TO
        table = TableSchemaFull(
            name="habit_events",
            columns=[
                _col("id", "uuid", is_pk=True),
                _col("user_id", "uuid"),
                _col("habit_template_id", "uuid"),
            ],
            foreign_keys=[
                _fk("habit_events", "user_id", "users"),
                _fk("habit_events", "habit_template_id", "habit_templates", on_delete="CASCADE"),
            ],
        )

        fk_user = table.foreign_keys[0]
        fk_template = table.foreign_keys[1]

        assert classify_fk_relation(fk_user) == RelationType.BELONGS_TO.value
        assert classify_fk_relation(fk_template, table) == RelationType.INSTANCE_OF.value

    def test_cascade_part_of(self):
        """CASCADE FK without template pattern → PART_OF."""
        fk = _fk("lessons", "course_id", "courses", on_delete="CASCADE")
        # course_id doesn't match ownership or template patterns
        assert classify_fk_relation(fk) == RelationType.PART_OF.value

    def test_junction_table_uses(self):
        """M2M junction table → USES."""
        table = TableSchemaFull(
            name="movement_muscle_links",
            columns=[
                _col("movement_id", "uuid", is_pk=True),
                _col("muscle_id", "uuid", is_pk=True),
                _col("role", "text"),
            ],
            foreign_keys=[
                _fk("movement_muscle_links", "movement_id", "movements"),
                _fk("movement_muscle_links", "muscle_id", "muscles"),
            ],
        )

        fk = table.foreign_keys[0]
        assert classify_fk_relation(fk, table) == RelationType.USES.value

    def test_catalog_detection(self):
        """Catalog detection via slug, is_active, template suffix."""
        # Has slug → catalog
        with_slug = TableSchemaFull(
            name="movements",
            columns=[_col("id", is_pk=True), _col("name"), _col("slug")],
        )
        assert detect_catalog_table(with_slug) is True

        # Has is_active → catalog
        with_active = TableSchemaFull(
            name="exercises",
            columns=[_col("id", is_pk=True), _col("name"), _col("is_active")],
        )
        assert detect_catalog_table(with_active) is True

        # Template suffix → catalog
        template = TableSchemaFull(
            name="habit_templates",
            columns=[_col("id", is_pk=True), _col("name")],
        )
        assert detect_catalog_table(template) is True

        # None of the above → not catalog
        plain = TableSchemaFull(
            name="habit_events",
            columns=[_col("id", is_pk=True), _col("date")],
        )
        assert detect_catalog_table(plain) is False

    def test_constraint_to_rule(self):
        """CHECK constraint → readable rule text."""
        check = CheckConstraintInfo(
            constraint_name="food_items_calories_check",
            table_name="food_items",
            check_clause="((calories >= 0))",
        )
        rule = constraint_to_rule(check, "food_items")
        assert "food_items" in rule.rule_text
        assert "calories >= 0" in rule.rule_text
        assert rule.category == "constraint"

    def test_domain_resolution(self):
        """auto_domain picks database_name over table prefix."""
        table = TableSchemaFull(name="habit_events", columns=[])
        assert auto_domain(table, "mindmirror") == "mindmirror"
        assert auto_domain(table, "") == "habit"

    def test_name_column_detection(self):
        """find_name_column finds name/title/slug."""
        table = TableSchemaFull(
            name="movements",
            columns=[_col("id"), _col("name"), _col("description")],
        )
        assert find_name_column(table) == "name"

        table2 = TableSchemaFull(
            name="courses",
            columns=[_col("id"), _col("title"), _col("slug")],
        )
        assert find_name_column(table2) == "title"


# ---------------------------------------------------------------------------
# TestGraphIngest
# ---------------------------------------------------------------------------


class TestGraphIngest:
    """Graph ingestion into InMemoryBackend."""

    @pytest.mark.asyncio
    async def test_creates_catalog_concepts(self):
        """Catalog table rows become ConceptNodes."""
        backend = InMemoryBackend()
        backend.connect()

        tables_data = {
            "movements": [
                {"id": "mv1", "name": "Squat", "slug": "squat", "description": "A squat"},
                {"id": "mv2", "name": "Deadlift", "slug": "deadlift", "description": "A deadlift"},
            ],
        }
        conn = _make_mock_conn(tables_data)

        schema = SchemaGraph(
            source_id="mm_movements",
            database_name="swae_movements",
            tables=[
                TableSchemaFull(
                    name="movements",
                    columns=[
                        _col("id", "uuid", is_pk=True),
                        _col("name", "text"),
                        _col("slug", "text"),
                        _col("description", "text"),
                    ],
                    row_count=2,
                ),
            ],
        )

        mapping = GraphMapping(
            tables=[
                TableMapping(
                    table_name="movements",
                    domain="exercise",
                    name_column="name",
                    description_columns=["description"],
                    is_catalog=True,
                ),
            ],
        )

        ingestor = PostgresGraphIngestor()
        counts = await ingestor.ingest(mapping, schema, conn, backend)

        assert counts["concepts"] == 2
        # Verify nodes exist in backend
        node = backend.get_node("mm_movements:movements:mv1")
        assert node is not None
        assert node.name == "Squat"
        assert node.domain == "exercise"

    @pytest.mark.asyncio
    async def test_creates_edges(self):
        """FK relationships create ConceptEdges."""
        backend = InMemoryBackend()
        backend.connect()

        conn = _make_mock_conn({
            "courses": [{"id": "c1", "title": "Spanish A1", "slug": "spanish-a1"}],
            "lessons": [{"id": "l1", "course_id": "c1", "title": "Greetings"}],
        })

        schema = SchemaGraph(
            source_id="interlinear",
            database_name="interlinear",
            tables=[
                TableSchemaFull(
                    name="courses",
                    columns=[_col("id", is_pk=True), _col("title"), _col("slug")],
                    row_count=1,
                ),
                TableSchemaFull(
                    name="lessons",
                    columns=[_col("id", is_pk=True), _col("course_id"), _col("title")],
                    foreign_keys=[_fk("lessons", "course_id", "courses", on_delete="CASCADE")],
                    row_count=1,
                ),
            ],
        )

        mapping = GraphMapping(
            tables=[
                TableMapping(table_name="courses", domain="language", name_column="title",
                             description_columns=[], is_catalog=True),
                TableMapping(table_name="lessons", domain="language", name_column="title",
                             description_columns=[], is_catalog=False),
            ],
            edges=[
                EdgeMapping(source_table="lessons", target_table="courses",
                            fk_column="course_id", relation_type=RelationType.PART_OF.value),
            ],
        )

        ingestor = PostgresGraphIngestor()
        counts = await ingestor.ingest(mapping, schema, conn, backend)

        assert counts["edges"] == 1
        # Verify the edge exists
        edges = list(backend.get_edges("interlinear:lessons:l1", "out"))
        assert len(edges) == 1
        assert edges[0].relation_type == RelationType.PART_OF

    @pytest.mark.asyncio
    async def test_creates_rules(self):
        """CHECK constraints become ExplicitRules."""
        backend = InMemoryBackend()
        backend.connect()

        conn = _make_mock_conn({
            "food_items": [{"id": "f1", "name": "Chicken", "calories": 165}],
        })

        schema = SchemaGraph(
            source_id="mm_main",
            database_name="mindmirror",
            tables=[
                TableSchemaFull(
                    name="food_items",
                    columns=[_col("id", is_pk=True), _col("name"), _col("calories")],
                    check_constraints=[
                        CheckConstraintInfo(
                            constraint_name="food_items_calories_check",
                            table_name="food_items",
                            check_clause="((calories >= 0))",
                        ),
                    ],
                    row_count=1,
                ),
            ],
        )

        mapping = GraphMapping(
            tables=[
                TableMapping(table_name="food_items", domain="nutrition",
                             name_column="name", description_columns=[], is_catalog=True),
            ],
            rules=[
                RuleMapping(
                    table_name="food_items",
                    constraint_name="food_items_calories_check",
                    rule_text="In food_items, calories >= 0",
                    category="constraint",
                ),
            ],
        )

        ingestor = PostgresGraphIngestor()
        counts = await ingestor.ingest(mapping, schema, conn, backend)

        assert counts["rules"] == 1

    @pytest.mark.asyncio
    async def test_embedding_catalog_rows(self):
        """Catalog rows get embedded when embedding model provided."""
        backend = InMemoryBackend()
        backend.connect()

        conn = _make_mock_conn({
            "movements": [{"id": "mv1", "name": "Squat", "slug": "squat"}],
        })

        schema = SchemaGraph(
            source_id="test",
            database_name="test",
            tables=[
                TableSchemaFull(
                    name="movements",
                    columns=[_col("id", is_pk=True), _col("name"), _col("slug")],
                    row_count=1,
                ),
            ],
        )

        mapping = GraphMapping(
            tables=[
                TableMapping(table_name="movements", domain="exercise",
                             name_column="name", is_catalog=True),
            ],
        )

        embedding = FakeEmbedding()
        ingestor = PostgresGraphIngestor()
        counts = await ingestor.ingest(mapping, schema, conn, backend, embedding)

        assert counts["concepts"] == 1
        # Verify embedding was stored
        emb = backend.get_embedding("test:movements:mv1")
        assert emb is not None
        assert len(emb) == 4

    @pytest.mark.asyncio
    async def test_skip_rules_when_disabled(self):
        """extract_rules=False skips rule creation."""
        from dataclasses import dataclass as _dc

        @_dc
        class _FakeIngestConfig:
            extract_rules: bool = False
            embed_catalog_tables: bool = True

        backend = InMemoryBackend()
        backend.connect()

        conn = _make_mock_conn({
            "items": [{"id": "i1", "name": "Test"}],
        })

        schema = SchemaGraph(
            source_id="test",
            database_name="test",
            tables=[
                TableSchemaFull(
                    name="items",
                    columns=[_col("id", is_pk=True), _col("name")],
                    row_count=1,
                ),
            ],
        )

        mapping = GraphMapping(
            tables=[TableMapping(table_name="items", domain="test", name_column="name")],
            rules=[
                RuleMapping(
                    table_name="items",
                    constraint_name="check_something",
                    rule_text="something",
                ),
            ],
        )

        ingestor = PostgresGraphIngestor(ingest_config=_FakeIngestConfig())
        counts = await ingestor.ingest(mapping, schema, conn, backend)

        assert counts["rules"] == 0


# ---------------------------------------------------------------------------
# TestCrossDB
# ---------------------------------------------------------------------------


class TestCrossDB:
    """Cross-database edge detection tests."""

    def test_movement_id_detection(self):
        """movement_id in practices → movements in another DB."""
        practices_schema = SchemaGraph(
            source_id="mm_practices",
            database_name="swae_practices",
            tables=[
                TableSchemaFull(
                    name="movement_templates",
                    columns=[
                        _col("id", "uuid", is_pk=True),
                        _col("movement_id", "uuid"),  # cross-DB, no FK
                    ],
                    foreign_keys=[],  # No FK — it's cross-DB
                ),
            ],
        )

        movements_schema = SchemaGraph(
            source_id="mm_movements",
            database_name="swae_movements",
            tables=[
                TableSchemaFull(
                    name="movements",
                    columns=[_col("id", "uuid", is_pk=True), _col("name")],
                ),
            ],
        )

        edges = detect_cross_db_edges_by_naming([practices_schema, movements_schema])
        assert len(edges) >= 1

        mv_edge = next(
            (e for e in edges if e.source_table == "movement_templates"
             and e.target_table == "movements"),
            None,
        )
        assert mv_edge is not None
        assert mv_edge.source_database == "swae_practices"
        assert mv_edge.target_database == "swae_movements"
        assert mv_edge.relation_type == RelationType.USES.value

    def test_user_identity_unification(self):
        """user_id across DBs → BELONGS_TO edges to users table."""
        users_schema = SchemaGraph(
            source_id="mm_users",
            database_name="swae_users",
            tables=[
                TableSchemaFull(
                    name="users",
                    columns=[_col("id", "uuid", is_pk=True), _col("email")],
                ),
            ],
        )

        main_schema = SchemaGraph(
            source_id="mm_main",
            database_name="mindmirror",
            tables=[
                TableSchemaFull(
                    name="habit_events",
                    columns=[_col("id", is_pk=True), _col("user_id", "uuid")],
                ),
                TableSchemaFull(
                    name="meals",
                    columns=[_col("id", is_pk=True), _col("user_id", "uuid")],
                ),
            ],
        )

        edges = create_user_identity_edges([users_schema, main_schema])
        assert len(edges) == 2

        for edge in edges:
            assert edge.target_table == "users"
            assert edge.target_database == "swae_users"
            assert edge.relation_type == RelationType.BELONGS_TO.value

    def test_no_users_table_returns_empty(self):
        """No users table → no identity edges."""
        schema = SchemaGraph(
            source_id="test",
            database_name="test",
            tables=[
                TableSchemaFull(
                    name="things",
                    columns=[_col("id", is_pk=True), _col("user_id")],
                ),
            ],
        )
        edges = create_user_identity_edges([schema])
        assert edges == []

    def test_discover_all_cross_db(self):
        """discover_all_cross_db combines naming + identity detection."""
        from qortex.sources.cross_db import discover_all_cross_db

        users_schema = SchemaGraph(
            source_id="users",
            database_name="users_db",
            tables=[
                TableSchemaFull(
                    name="users",
                    columns=[_col("id", "uuid", is_pk=True)],
                ),
            ],
        )

        practices_schema = SchemaGraph(
            source_id="practices",
            database_name="practices_db",
            tables=[
                TableSchemaFull(
                    name="sessions",
                    columns=[
                        _col("id", is_pk=True),
                        _col("user_id", "uuid"),
                        _col("movement_id", "uuid"),
                    ],
                    foreign_keys=[],
                ),
            ],
        )

        movements_schema = SchemaGraph(
            source_id="movements",
            database_name="movements_db",
            tables=[
                TableSchemaFull(
                    name="movements",
                    columns=[_col("id", "uuid", is_pk=True), _col("name")],
                ),
            ],
        )

        edges = discover_all_cross_db([users_schema, practices_schema, movements_schema])
        # Should find: movement_id → movements (naming) + user_id → users (identity)
        assert len(edges) >= 2

        has_movement = any(
            e.source_column == "movement_id" and e.target_table == "movements"
            for e in edges
        )
        has_user = any(
            e.source_column == "user_id" and e.target_table == "users"
            for e in edges
        )
        assert has_movement
        assert has_user


# ---------------------------------------------------------------------------
# TestEndToEnd
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """End-to-end: discover → map → ingest."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Full pipeline: discover → map → ingest with real InMemoryBackend."""
        backend = InMemoryBackend()
        backend.connect()

        tables_data = {
            "habit_templates": [
                {"id": "ht1", "name": "Meditation", "slug": "meditation",
                 "description": "Daily meditation practice"},
            ],
            "habit_events": [
                {"id": "he1", "user_id": "u1", "habit_template_id": "ht1",
                 "date": "2026-02-01", "response": "completed"},
            ],
        }
        conn = _make_mock_conn(tables_data)

        ingestor = PostgresGraphIngestor()
        schema = await ingestor.discover_schema(
            conn, database_name="mindmirror", source_id="mm_main"
        )

        assert len(schema.tables) == 2

        mapping = ingestor.map_schema(schema)
        assert len(mapping.tables) == 2

        # Verify catalog detection
        ht_mapping = next(m for m in mapping.tables if m.table_name == "habit_templates")
        assert ht_mapping.is_catalog is True
        assert ht_mapping.name_column == "name"

        counts = await ingestor.ingest(mapping, schema, conn, backend)
        assert counts["concepts"] >= 2  # At least both tables' rows

    @pytest.mark.asyncio
    async def test_run_convenience(self):
        """run() method chains discover → map → ingest."""
        backend = InMemoryBackend()
        backend.connect()

        conn = _make_mock_conn({
            "movements": [
                {"id": "mv1", "name": "Squat", "slug": "squat"},
            ],
        })

        from dataclasses import dataclass as _dc, field as _field

        @_dc
        class _FakeConfig:
            source_id: str = "mm_movements"
            connection_string: str = "pg://test"
            schemas: list[str] = _field(default_factory=lambda: ["public"])
            domain_map: dict[str, str] = _field(default_factory=lambda: {"*": "exercise"})

        config = _FakeConfig()

        ingestor = PostgresGraphIngestor(config=config, backend=backend)
        counts = await ingestor.run(conn=conn)

        assert counts["concepts"] >= 1
        node = backend.get_node("mm_movements:movements:mv1")
        assert node is not None
        assert node.name == "Squat"


# ---------------------------------------------------------------------------
# TestMCPGraphTools
# ---------------------------------------------------------------------------


class TestMCPGraphTools:
    """MCP tool _impl functions for source graph ingestion.

    These test the graph-level integration without requiring asyncpg.
    We test at the PostgresGraphIngestor level with mock connections.
    """

    @pytest.mark.asyncio
    async def test_ingest_via_ingestor_creates_nodes_in_backend(self):
        """Full ingestor pipeline writes nodes to real InMemoryBackend."""
        backend = InMemoryBackend()
        backend.connect()

        conn = _make_mock_conn({
            "movements": [
                {"id": "mv1", "name": "Squat", "slug": "squat"},
                {"id": "mv2", "name": "Deadlift", "slug": "deadlift"},
            ],
        })

        from dataclasses import dataclass as _dc, field as _field

        @_dc
        class FakeConfig:
            source_id: str = "mm_movements"
            schemas: list[str] = _field(default_factory=lambda: ["public"])
            domain_map: dict[str, str] = _field(default_factory=lambda: {"*": "exercise"})

        ingestor = PostgresGraphIngestor(
            config=FakeConfig(), backend=backend, embedding_model=FakeEmbedding()
        )
        counts = await ingestor.run(conn=conn)

        assert counts["concepts"] == 2
        assert counts["edges"] == 0
        assert counts["rules"] == 0

        node = backend.get_node("mm_movements:movements:mv1")
        assert node is not None
        assert node.name == "Squat"
        assert node.domain == "exercise"

        # Catalog rows should be embedded
        emb = backend.get_embedding("mm_movements:movements:mv1")
        assert emb is not None

    @pytest.mark.asyncio
    async def test_ingest_with_edges_and_rules(self):
        """Ingestor creates edges and rules from FKs and CHECK constraints."""
        backend = InMemoryBackend()
        backend.connect()

        # Build schema + mapping manually to test ingest() directly
        schema = SchemaGraph(
            source_id="mm_main",
            database_name="mindmirror",
            tables=[
                TableSchemaFull(
                    name="habit_templates",
                    columns=[
                        _col("id", "uuid", is_pk=True),
                        _col("name", "text"),
                        _col("slug", "text"),
                    ],
                    row_count=1,
                ),
                TableSchemaFull(
                    name="habit_events",
                    columns=[
                        _col("id", "uuid", is_pk=True),
                        _col("habit_template_id", "uuid"),
                    ],
                    foreign_keys=[
                        _fk("habit_events", "habit_template_id", "habit_templates",
                             on_delete="CASCADE"),
                    ],
                    check_constraints=[
                        CheckConstraintInfo(
                            constraint_name="events_check",
                            table_name="habit_events",
                            check_clause="((id IS NOT NULL))",
                        ),
                    ],
                    row_count=1,
                ),
            ],
        )

        conn = _make_mock_conn({
            "habit_templates": [{"id": "ht1", "name": "Meditation", "slug": "meditation"}],
            "habit_events": [{"id": "he1", "habit_template_id": "ht1"}],
        })

        ingestor = PostgresGraphIngestor(backend=backend)
        mapping = ingestor.map_schema(schema)
        counts = await ingestor.ingest(mapping, schema, conn, backend)

        assert counts["concepts"] == 2
        assert counts["edges"] == 1
        assert counts["rules"] >= 1

        # Verify edge
        edges = list(backend.get_edges("mm_main:habit_events:he1", "out"))
        assert len(edges) == 1
        assert edges[0].target_id == "mm_main:habit_templates:ht1"

    def test_map_schema_returns_correct_structure(self):
        """map_schema() produces TableMapping, EdgeMapping, RuleMapping."""
        schema = SchemaGraph(
            source_id="test",
            database_name="test_db",
            tables=[
                TableSchemaFull(
                    name="items",
                    columns=[_col("id", is_pk=True), _col("name"), _col("slug")],
                    check_constraints=[
                        CheckConstraintInfo(
                            constraint_name="items_check",
                            table_name="items",
                            check_clause="((name IS NOT NULL))",
                        ),
                    ],
                ),
            ],
        )

        ingestor = PostgresGraphIngestor()
        mapping = ingestor.map_schema(schema)

        assert len(mapping.tables) == 1
        assert mapping.tables[0].is_catalog is True  # has slug
        assert mapping.tables[0].name_column == "name"
        assert len(mapping.rules) == 1
        assert "items" in mapping.rules[0].rule_text
