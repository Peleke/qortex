"""Tests for Layer 2 core: source adapter types, serializers, registry.

No database drivers needed — pure types and serialization logic.
"""

from __future__ import annotations

import pytest

from qortex.sources.base import ColumnSchema, SourceConfig, SyncResult, TableSchema
from qortex.sources.registry import SourceRegistry
from qortex.sources.serializer import (
    KeyValueSerializer,
    NaturalLanguageSerializer,
    _format_value,
    _humanize_column_name,
    _humanize_table_name,
    _is_internal_column,
)

# ===========================================================================
# Schema types
# ===========================================================================


class TestColumnSchema:
    def test_basic_column(self):
        col = ColumnSchema(name="name", data_type="text")
        assert col.name == "name"
        assert col.data_type == "text"
        assert col.nullable is True
        assert col.is_pk is False
        assert col.is_fk is False

    def test_pk_column(self):
        col = ColumnSchema(name="id", data_type="uuid", is_pk=True, nullable=False)
        assert col.is_pk is True
        assert col.nullable is False

    def test_fk_column(self):
        col = ColumnSchema(
            name="user_id",
            data_type="uuid",
            is_fk=True,
            foreign_table="users",
            foreign_column="id",
        )
        assert col.is_fk is True
        assert col.foreign_table == "users"
        assert col.foreign_column == "id"


class TestTableSchema:
    def test_basic_table(self):
        cols = [
            ColumnSchema(name="id", data_type="uuid", is_pk=True),
            ColumnSchema(name="name", data_type="text"),
        ]
        table = TableSchema(name="users", columns=cols, row_count=100)
        assert table.name == "users"
        assert table.row_count == 100
        assert len(table.columns) == 2

    def test_pk_columns(self):
        cols = [
            ColumnSchema(name="id", data_type="uuid", is_pk=True),
            ColumnSchema(name="name", data_type="text"),
            ColumnSchema(name="version", data_type="integer", is_pk=True),
        ]
        table = TableSchema(name="versioned", columns=cols)
        assert table.pk_columns == ["id", "version"]

    def test_fk_columns(self):
        cols = [
            ColumnSchema(name="id", data_type="uuid", is_pk=True),
            ColumnSchema(name="user_id", data_type="uuid", is_fk=True, foreign_table="users"),
            ColumnSchema(name="name", data_type="text"),
        ]
        table = TableSchema(name="items", columns=cols)
        assert len(table.fk_columns) == 1
        assert table.fk_columns[0].name == "user_id"

    def test_full_name(self):
        table = TableSchema(name="users", schema_name="auth")
        assert table.full_name == "auth.users"

    def test_default_schema(self):
        table = TableSchema(name="users")
        assert table.full_name == "public.users"


class TestSyncResult:
    def test_defaults(self):
        result = SyncResult(source_id="test")
        assert result.tables_synced == 0
        assert result.rows_added == 0
        assert result.errors == []

    def test_with_data(self):
        result = SyncResult(
            source_id="mm",
            tables_synced=4,
            rows_added=800,
            vectors_created=800,
            duration_seconds=12.5,
        )
        assert result.tables_synced == 4
        assert result.vectors_created == 800


# ===========================================================================
# SourceConfig
# ===========================================================================


class TestSourceConfig:
    def test_basic_config(self):
        config = SourceConfig(
            source_id="mm_movements",
            connection_string="postgresql://localhost:5435/swae_movements",
        )
        assert config.source_id == "mm_movements"
        assert config.batch_size == 500
        assert config.default_domain == "default"

    def test_resolve_domain_exact_match(self):
        config = SourceConfig(
            source_id="mm",
            connection_string="postgresql://localhost/mindmirror",
            domain_map={
                "habit_*": "habits",
                "food_*": "nutrition",
                "journal_*": "journal",
            },
        )
        assert config.resolve_domain("habit_templates") == "habits"
        assert config.resolve_domain("food_items") == "nutrition"
        assert config.resolve_domain("journal_entries") == "journal"

    def test_resolve_domain_wildcard(self):
        config = SourceConfig(
            source_id="mm",
            connection_string="postgresql://localhost/mindmirror",
            domain_map={"*": "catchall"},
        )
        assert config.resolve_domain("anything") == "catchall"

    def test_resolve_domain_falls_back_to_default(self):
        config = SourceConfig(
            source_id="mm",
            connection_string="postgresql://localhost/mindmirror",
            domain_map={"habit_*": "habits"},
            default_domain="misc",
        )
        assert config.resolve_domain("unknown_table") == "misc"

    def test_exclude_tables(self):
        config = SourceConfig(
            source_id="mm",
            connection_string="postgresql://localhost/mindmirror",
            exclude_tables=["_prisma_migrations", "schema_migrations"],
        )
        assert "_prisma_migrations" in config.exclude_tables


# ===========================================================================
# Serializer helpers
# ===========================================================================


class TestSerializerHelpers:
    def test_is_internal_column(self):
        assert _is_internal_column("id") is True
        assert _is_internal_column("uuid") is True
        assert _is_internal_column("created_at") is True
        assert _is_internal_column("user_id") is True
        assert _is_internal_column("name") is False
        assert _is_internal_column("calories") is False
        assert _is_internal_column("description") is False

    def test_humanize_table_name(self):
        assert _humanize_table_name("food_items") == "food item"
        assert _humanize_table_name("users") == "user"
        assert _humanize_table_name("address") == "address"
        assert _humanize_table_name("habit_templates") == "habit template"

    def test_humanize_column_name(self):
        assert _humanize_column_name("max_heart_rate") == "max heart rate"
        assert _humanize_column_name("name") == "name"

    def test_format_value_none(self):
        assert _format_value(None) == ""

    def test_format_value_bool(self):
        assert _format_value(True) == "yes"
        assert _format_value(False) == "no"

    def test_format_value_float(self):
        assert _format_value(3.0) == "3"
        assert _format_value(3.14) == "3.14"

    def test_format_value_list(self):
        assert _format_value(["a", "b", "c"]) == "a, b, c"

    def test_format_value_dict(self):
        result = _format_value({"mood": "happy", "energy": 8})
        assert "mood: happy" in result
        assert "energy: 8" in result


# ===========================================================================
# NaturalLanguageSerializer
# ===========================================================================


class TestNaturalLanguageSerializer:
    def test_simple_row(self):
        s = NaturalLanguageSerializer()
        result = s.serialize(
            "food_items",
            {"name": "Chicken Breast", "calories": 165, "protein": 31},
        )
        assert "food item" in result
        assert "Chicken Breast" in result
        assert "calories" in result
        assert "165" in result

    def test_skips_internal_columns(self):
        s = NaturalLanguageSerializer()
        result = s.serialize(
            "users",
            {"id": "abc-123", "name": "Alice", "created_at": "2024-01-01"},
        )
        assert "abc-123" not in result
        assert "Alice" in result
        assert "2024-01-01" not in result

    def test_prioritizes_name(self):
        s = NaturalLanguageSerializer()
        result = s.serialize(
            "movements",
            {"name": "Deadlift", "muscle_group": "posterior chain", "difficulty": "advanced"},
        )
        assert result.startswith("A movement named 'Deadlift'")

    def test_includes_description(self):
        s = NaturalLanguageSerializer()
        result = s.serialize(
            "habits",
            {"name": "Morning Run", "description": "Run 5km every morning before breakfast"},
        )
        assert "Morning Run" in result
        assert "Run 5km" in result

    def test_mindmirror_food_item(self):
        """Realistic MindMirror food_items row."""
        s = NaturalLanguageSerializer()
        result = s.serialize(
            "food_items",
            {
                "id": "uuid-123",
                "name": "Salmon Fillet",
                "calories": 208,
                "protein": 20.4,
                "fat": 13.4,
                "carbs": 0,
                "serving_size": "100g",
                "created_at": "2024-01-01",
            },
        )
        assert "food item" in result
        assert "Salmon Fillet" in result
        assert "208" in result
        assert "uuid-123" not in result
        assert "2024-01-01" not in result

    def test_mindmirror_movement(self):
        """Realistic MindMirror movement row."""
        s = NaturalLanguageSerializer()
        result = s.serialize(
            "movements",
            {
                "id": 42,
                "name": "Barbell Back Squat",
                "description": "A compound lower body exercise targeting quads and glutes",
                "muscle_group": "quadriceps",
                "equipment": "barbell",
                "difficulty": "intermediate",
            },
        )
        assert "Barbell Back Squat" in result
        assert "compound lower body" in result
        assert "quadriceps" in result

    def test_max_length(self):
        s = NaturalLanguageSerializer(max_length=50)
        result = s.serialize(
            "items",
            {"name": "X" * 100, "description": "Y" * 100},
        )
        assert len(result) <= 50

    def test_none_values_skipped(self):
        s = NaturalLanguageSerializer()
        result = s.serialize(
            "items",
            {"name": "Test", "optional_field": None, "empty": ""},
        )
        assert "optional_field" not in result
        assert "empty" not in result

    def test_skip_internal_disabled(self):
        s = NaturalLanguageSerializer(skip_internal=False)
        result = s.serialize(
            "items",
            {"id": "123", "name": "Test", "created_at": "2024-01-01"},
        )
        assert "123" in result


# ===========================================================================
# KeyValueSerializer
# ===========================================================================


class TestKeyValueSerializer:
    def test_simple_row(self):
        s = KeyValueSerializer()
        result = s.serialize(
            "food_items",
            {"name": "Rice", "calories": 130},
        )
        assert "table=food_items" in result
        assert "name=Rice" in result
        assert "calories=130" in result

    def test_skips_internal(self):
        s = KeyValueSerializer()
        result = s.serialize(
            "items",
            {"id": "abc", "name": "Test", "user_id": "xyz"},
        )
        assert "id=" not in result
        assert "user_id=" not in result
        assert "name=Test" in result

    def test_custom_separator(self):
        s = KeyValueSerializer(separator=" | ")
        result = s.serialize("t", {"name": "A", "val": 1})
        assert " | " in result


# ===========================================================================
# SourceRegistry
# ===========================================================================


class TestSourceRegistry:
    def test_register_and_get(self):
        registry = SourceRegistry()
        config = SourceConfig(source_id="test", connection_string="mock://")
        adapter = object()
        registry.register(config, adapter)

        assert registry.get("test") is adapter
        assert registry.get_config("test") is config

    def test_get_missing_returns_none(self):
        registry = SourceRegistry()
        assert registry.get("missing") is None
        assert registry.get_config("missing") is None

    def test_remove(self):
        registry = SourceRegistry()
        config = SourceConfig(source_id="test", connection_string="mock://")
        registry.register(config, object())

        assert registry.remove("test") is True
        assert registry.get("test") is None
        assert registry.remove("test") is False

    @pytest.mark.asyncio
    async def test_remove_async_disconnects(self):
        registry = SourceRegistry()
        config = SourceConfig(source_id="test", connection_string="mock://")

        class FakeAdapter:
            disconnected = False

            async def disconnect(self):
                self.disconnected = True

        adapter = FakeAdapter()
        registry.register(config, adapter)

        assert await registry.remove_async("test") is True
        assert adapter.disconnected is True
        assert registry.get("test") is None

    def test_list_sources(self):
        registry = SourceRegistry()
        registry.register(SourceConfig(source_id="a", connection_string=""), object())
        registry.register(SourceConfig(source_id="b", connection_string=""), object())

        sources = registry.list_sources()
        assert set(sources) == {"a", "b"}

    def test_cache_schemas(self):
        registry = SourceRegistry()
        config = SourceConfig(source_id="test", connection_string="")
        registry.register(config, object())

        schemas = [TableSchema(name="users"), TableSchema(name="items")]
        registry.cache_schemas("test", schemas)

        cached = registry.get_schemas("test")
        assert cached is not None
        assert len(cached) == 2

    def test_clear(self):
        registry = SourceRegistry()
        registry.register(SourceConfig(source_id="a", connection_string=""), object())
        registry.register(SourceConfig(source_id="b", connection_string=""), object())

        registry.clear()
        assert registry.list_sources() == []

    @pytest.mark.asyncio
    async def test_clear_async_disconnects(self):
        registry = SourceRegistry()

        class FakeAdapter:
            disconnected = False

            async def disconnect(self):
                self.disconnected = True

        a1, a2 = FakeAdapter(), FakeAdapter()
        registry.register(SourceConfig(source_id="a", connection_string=""), a1)
        registry.register(SourceConfig(source_id="b", connection_string=""), a2)

        await registry.clear_async()
        assert registry.list_sources() == []
        assert a1.disconnected is True
        assert a2.disconnected is True
