"""Tests for Layer 3 core: graph mapping types + mechanical rules.

Tests FK classification, constraint→rule, column detection, catalog detection,
cross-database edge detection.
"""

from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Helpers: build MindMirror-shaped schemas
# ---------------------------------------------------------------------------


def _col(name: str, data_type: str = "text", is_pk: bool = False, **kwargs) -> dict:
    return {"name": name, "data_type": data_type, "is_pk": is_pk, **kwargs}


def _fk(
    source_table: str,
    source_column: str,
    target_table: str,
    target_column: str = "id",
    on_delete: str = "NO ACTION",
    constraint_name: str = "",
) -> ForeignKeyInfo:
    return ForeignKeyInfo(
        constraint_name=constraint_name or f"fk_{source_table}_{source_column}",
        source_table=source_table,
        source_column=source_column,
        target_table=target_table,
        target_column=target_column,
        on_delete=on_delete,
    )


# ===========================================================================
# FK Classification
# ===========================================================================


class TestClassifyFKRelation:
    def test_user_id_is_belongs_to(self):
        fk = _fk("meals", "user_id", "users")
        assert classify_fk_relation(fk) == RelationType.BELONGS_TO.value

    def test_owner_id_is_belongs_to(self):
        fk = _fk("documents", "owner_id", "users")
        assert classify_fk_relation(fk) == RelationType.BELONGS_TO.value

    def test_template_id_is_instance_of(self):
        fk = _fk("habit_events", "habit_template_id", "habit_templates")
        assert classify_fk_relation(fk) == RelationType.INSTANCE_OF.value

    def test_category_id_is_instance_of(self):
        fk = _fk("products", "product_category_id", "categories")
        assert classify_fk_relation(fk) == RelationType.INSTANCE_OF.value

    def test_cascade_delete_is_part_of(self):
        fk = _fk(
            "prescription_instances",
            "practice_instance_id",
            "practice_instances",
            on_delete="CASCADE",
        )
        assert classify_fk_relation(fk) == RelationType.PART_OF.value

    def test_junction_table_is_uses(self):
        """M2M junction tables → USES."""
        table = TableSchemaFull(
            name="movement_muscle_links",
            columns=[
                _col("movement_id", "uuid", is_pk=True),
                _col("muscle_name", "text", is_pk=True),
                _col("role", "text", is_pk=True),
            ],
            foreign_keys=[
                _fk("movement_muscle_links", "movement_id", "movements"),
            ],
        )
        fk = table.foreign_keys[0]
        result = classify_fk_relation(fk, source_table=table)
        assert result == RelationType.USES.value

    def test_junction_table_suffix_links(self):
        table = TableSchemaFull(
            name="movement_equipment_links",
            columns=[
                _col("movement_id", "uuid"),
                _col("equipment_name", "text"),
                _col("role", "text"),
                _col("item_count", "integer"),
            ],
            foreign_keys=[
                _fk("movement_equipment_links", "movement_id", "movements"),
            ],
        )
        fk = table.foreign_keys[0]
        assert classify_fk_relation(fk, source_table=table) == RelationType.USES.value

    def test_generic_fk_is_part_of(self):
        fk = _fk("items", "category_ref", "categories")
        assert classify_fk_relation(fk) == RelationType.PART_OF.value


# ===========================================================================
# Constraint → Rule
# ===========================================================================


class TestConstraintToRule:
    def test_simple_check(self):
        constraint = CheckConstraintInfo(
            constraint_name="ck_calories_positive",
            table_name="food_items",
            check_clause="((calories >= 0))",
        )
        rule = constraint_to_rule(constraint, "food_items")
        assert "food_items" in rule.rule_text
        assert "calories" in rule.rule_text
        assert rule.category == "constraint"

    def test_enum_check(self):
        constraint = CheckConstraintInfo(
            constraint_name="ck_status",
            table_name="enrollments",
            check_clause="((status = ANY (ARRAY['active', 'inactive', 'completed'])))",
        )
        rule = constraint_to_rule(constraint, "enrollments")
        assert "enrollments" in rule.rule_text
        assert "status" in rule.rule_text

    def test_nested_parens_stripped(self):
        constraint = CheckConstraintInfo(
            constraint_name="ck_test",
            table_name="t",
            check_clause="((x > 0))",
        )
        rule = constraint_to_rule(constraint, "t")
        assert rule.rule_text == "In t, x > 0"


# ===========================================================================
# Column Detection
# ===========================================================================


class TestFindNameColumn:
    def test_finds_name(self):
        table = TableSchemaFull(
            name="movements",
            columns=[_col("id", "uuid"), _col("name", "text"), _col("slug", "text")],
        )
        assert find_name_column(table) == "name"

    def test_finds_title(self):
        table = TableSchemaFull(
            name="programs",
            columns=[_col("id", "uuid"), _col("title", "text"), _col("description", "text")],
        )
        assert find_name_column(table) == "title"

    def test_prefers_name_over_title(self):
        table = TableSchemaFull(
            name="items",
            columns=[_col("name", "text"), _col("title", "text")],
        )
        assert find_name_column(table) == "name"

    def test_returns_none_when_no_match(self):
        table = TableSchemaFull(
            name="events",
            columns=[_col("id", "uuid"), _col("date", "date"), _col("response", "text")],
        )
        assert find_name_column(table) is None


class TestFindDescriptionColumns:
    def test_finds_description(self):
        table = TableSchemaFull(
            name="movements",
            columns=[_col("name", "text"), _col("description", "text")],
        )
        assert "description" in find_description_columns(table)

    def test_finds_multiple(self):
        table = TableSchemaFull(
            name="lessons",
            columns=[
                _col("title", "text"),
                _col("description", "text"),
                _col("markdown_content", "text"),
                _col("summary", "text"),
            ],
        )
        result = find_description_columns(table)
        assert "description" in result
        assert "markdown_content" in result
        assert "summary" in result


# ===========================================================================
# Auto Domain
# ===========================================================================


class TestAutoDomain:
    def test_uses_schema_name(self):
        table = TableSchemaFull(name="movements", schema_name="exercise")
        assert auto_domain(table) == "exercise"

    def test_falls_back_to_database_name(self):
        table = TableSchemaFull(name="movements", schema_name="public")
        assert auto_domain(table, database_name="swae_movements") == "swae_movements"

    def test_extracts_table_prefix(self):
        table = TableSchemaFull(name="habit_templates", schema_name="public")
        assert auto_domain(table) == "habit"

    def test_returns_table_name_for_single_word(self):
        table = TableSchemaFull(name="users", schema_name="public")
        assert auto_domain(table) == "users"


# ===========================================================================
# Catalog Detection
# ===========================================================================


class TestDetectCatalogTable:
    def test_slug_means_catalog(self):
        table = TableSchemaFull(
            name="movements",
            columns=[_col("id", "uuid"), _col("slug", "text"), _col("name", "text")],
        )
        assert detect_catalog_table(table) is True

    def test_templates_suffix_means_catalog(self):
        table = TableSchemaFull(name="habit_templates", columns=[_col("id", "uuid")])
        assert detect_catalog_table(table) is True

    def test_items_suffix_means_catalog(self):
        table = TableSchemaFull(name="food_items", columns=[_col("id", "uuid")])
        assert detect_catalog_table(table) is True

    def test_is_active_means_catalog(self):
        table = TableSchemaFull(
            name="programs",
            columns=[_col("id", "uuid"), _col("is_active", "boolean")],
        )
        assert detect_catalog_table(table) is True

    def test_event_table_is_not_catalog(self):
        table = TableSchemaFull(
            name="habit_events",
            columns=[
                _col("id", "uuid"),
                _col("user_id", "text"),
                _col("date", "date"),
                _col("response", "text"),
            ],
        )
        assert detect_catalog_table(table) is False


# ===========================================================================
# Cross-Database Edge Detection
# ===========================================================================


class TestCrossDBEdges:
    def _make_mindmirror_schemas(self) -> list[SchemaGraph]:
        """Build simplified MindMirror multi-DB schemas for testing."""
        movements = SchemaGraph(
            source_id="mm_movements",
            database_name="swae_movements",
            tables=[
                TableSchemaFull(
                    name="movements",
                    columns=[
                        _col("id", "uuid", is_pk=True),
                        _col("name", "text"),
                    ],
                ),
            ],
        )
        practices = SchemaGraph(
            source_id="mm_practices",
            database_name="swae_practices",
            tables=[
                TableSchemaFull(
                    name="movement_templates",
                    columns=[
                        _col("id", "uuid", is_pk=True),
                        _col("movement_id", "uuid"),  # Cross-DB ref to movements
                        _col(
                            "exercise_id", "uuid"
                        ),  # Also cross-DB but name doesn't match any table
                        _col("prescription_template_id", "uuid"),
                    ],
                    foreign_keys=[
                        # prescription_template_id is a local FK
                        _fk(
                            "movement_templates",
                            "prescription_template_id",
                            "prescription_templates",
                        ),
                    ],
                ),
                TableSchemaFull(
                    name="practice_instances",
                    columns=[
                        _col("id", "uuid", is_pk=True),
                        _col("user_id", "uuid"),
                    ],
                ),
            ],
        )
        users = SchemaGraph(
            source_id="mm_users",
            database_name="swae_users",
            tables=[
                TableSchemaFull(
                    name="users",
                    columns=[
                        _col("id", "uuid", is_pk=True),
                        _col("email", "text"),
                    ],
                ),
            ],
        )
        return [movements, practices, users]

    def test_detect_movement_id_cross_ref(self):
        """movement_id in practices → movements table in movements DB."""
        schemas = self._make_mindmirror_schemas()
        edges = detect_cross_db_edges_by_naming(schemas)

        movement_edges = [e for e in edges if e.source_column == "movement_id"]
        assert len(movement_edges) >= 1
        assert movement_edges[0].source_table == "movement_templates"
        assert movement_edges[0].target_database == "swae_movements"
        assert movement_edges[0].target_table == "movements"

    def test_user_identity_edges(self):
        """user_id columns across DBs → users table."""
        schemas = self._make_mindmirror_schemas()
        edges = create_user_identity_edges(schemas)

        # practice_instances.user_id → users.id
        pi_edges = [e for e in edges if e.source_table == "practice_instances"]
        assert len(pi_edges) == 1
        assert pi_edges[0].target_table == "users"
        assert pi_edges[0].relation_type == RelationType.BELONGS_TO.value

    def test_user_identity_skips_same_db(self):
        """Should not create edges within the same database."""
        schemas = self._make_mindmirror_schemas()
        edges = create_user_identity_edges(schemas)

        same_db = [e for e in edges if e.source_database == "swae_users"]
        assert len(same_db) == 0

    def test_no_cross_db_for_local_fk(self):
        """Local FKs should not be detected as cross-DB edges."""
        schemas = self._make_mindmirror_schemas()
        edges = detect_cross_db_edges_by_naming(schemas)

        # prescription_template_id is a local FK, should not appear
        local = [e for e in edges if "prescription" in e.source_column.lower()]
        assert len(local) == 0


# ===========================================================================
# Graph Mapping Types
# ===========================================================================


class TestGraphMappingTypes:
    def test_table_mapping(self):
        m = TableMapping(
            table_name="movements",
            domain="exercise",
            name_column="name",
            description_columns=["description"],
            is_catalog=True,
        )
        assert m.table_name == "movements"
        assert m.is_catalog is True

    def test_edge_mapping(self):
        m = EdgeMapping(
            source_table="habit_events",
            target_table="habit_templates",
            fk_column="habit_template_id",
            relation_type=RelationType.INSTANCE_OF.value,
        )
        assert m.relation_type == "instance_of"

    def test_graph_mapping_aggregation(self):
        gm = GraphMapping(
            tables=[TableMapping(table_name="t1", domain="d1")],
            edges=[
                EdgeMapping(
                    source_table="t1", target_table="t2", fk_column="t2_id", relation_type="uses"
                )
            ],
            rules=[RuleMapping(table_name="t1", constraint_name="ck", rule_text="rule")],
            cross_db_edges=[
                CrossDatabaseEdge(
                    source_database="a",
                    source_table="t1",
                    source_column="x_id",
                    target_database="b",
                    target_table="t2",
                    target_column="id",
                    relation_type="uses",
                )
            ],
        )
        assert len(gm.tables) == 1
        assert len(gm.edges) == 1
        assert len(gm.rules) == 1
        assert len(gm.cross_db_edges) == 1


class TestTableSchemaFull:
    def test_pk_columns(self):
        t = TableSchemaFull(
            name="test",
            columns=[
                _col("id", "uuid", is_pk=True),
                _col("name", "text"),
            ],
        )
        assert t.pk_columns == ["id"]

    def test_full_name(self):
        t = TableSchemaFull(name="users", schema_name="auth")
        assert t.full_name == "auth.users"

    def test_column_by_name(self):
        t = TableSchemaFull(
            name="test",
            columns=[_col("id", "uuid"), _col("name", "text")],
        )
        assert t.column_by_name("name")["data_type"] == "text"
        assert t.column_by_name("missing") is None
