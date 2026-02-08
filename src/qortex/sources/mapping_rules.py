"""Mechanical mapping rules: FK → edge type, constraint → rule text, schema heuristics.

These are deterministic, zero-LLM transformations from database schemas
to knowledge graph structure. The rules encode domain knowledge about
common database patterns.
"""

from __future__ import annotations

import re

from qortex.core.models import RelationType
from qortex.sources.graph_ingestor import (
    CheckConstraintInfo,
    CrossDatabaseEdge,
    ForeignKeyInfo,
    RuleMapping,
    SchemaGraph,
    TableSchemaFull,
)

# ---------------------------------------------------------------------------
# FK → RelationType classification
# ---------------------------------------------------------------------------

# Columns that indicate ownership (BELONGS_TO)
_OWNERSHIP_PATTERNS = {"user_id", "owner_id", "author_id", "creator_id", "created_by"}

# Columns that indicate template/catalog instantiation (INSTANCE_OF)
_TEMPLATE_PATTERNS = re.compile(
    r".*_template_id$|.*_type_id$|.*_category_id$|.*_class_id$|template_id$"
)

# Tables that are typically M2M junction tables
_JUNCTION_SUFFIXES = ("_links", "_associations", "_tags", "_roles")


def classify_fk_relation(fk: ForeignKeyInfo, source_table: TableSchemaFull | None = None) -> str:
    """Classify a foreign key into a RelationType value.

    Rules (in priority order):
    1. user_id, owner_id, etc. → BELONGS_TO
    2. Source table is a junction table (M2M) → USES
    3. CASCADE on delete → PART_OF (child lifecycle depends on parent)
    4. *_template_id, *_type_id → INSTANCE_OF (only if not CASCADE)
    5. Default → PART_OF
    """
    col_lower = fk.source_column.lower()

    # 1. Ownership
    if col_lower in _OWNERSHIP_PATTERNS:
        return RelationType.BELONGS_TO.value

    # 2. Junction table (M2M) — check before template pattern
    if source_table is not None:
        table_lower = source_table.name.lower()
        if any(table_lower.endswith(s) for s in _JUNCTION_SUFFIXES):
            return RelationType.USES.value
        # Also check: table has exactly 2 FK columns and no non-FK non-PK columns
        fk_cols = {f.source_column for f in source_table.foreign_keys}
        pk_cols = set(source_table.pk_columns)
        non_key_cols = [
            c["name"]
            for c in source_table.columns
            if c["name"] not in fk_cols and c["name"] not in pk_cols
            and c["name"] not in ("created_at", "modified_at", "id")
        ]
        if len(source_table.foreign_keys) >= 2 and len(non_key_cols) <= 2:
            return RelationType.USES.value

    # 3. CASCADE → tight coupling → PART_OF (overrides template pattern)
    if fk.on_delete == "CASCADE":
        return RelationType.PART_OF.value

    # 4. Template/catalog instantiation (only if not CASCADE)
    if _TEMPLATE_PATTERNS.match(col_lower):
        return RelationType.INSTANCE_OF.value

    # 5. Default
    return RelationType.PART_OF.value


# ---------------------------------------------------------------------------
# Constraint → Rule text
# ---------------------------------------------------------------------------


def constraint_to_rule(constraint: CheckConstraintInfo, table: str) -> RuleMapping:
    """Convert a CHECK constraint to a human-readable rule.

    Examples:
        ((calories >= 0)) → "In food_items, calories must be >= 0"
        ((status IN ('active', 'inactive'))) → "In enrollments, status must be one of: active, inactive"
    """
    clause = constraint.check_clause

    # Strip outer parens
    clean = clause.strip()
    while clean.startswith("(") and clean.endswith(")"):
        inner = clean[1:-1]
        # Only strip if balanced
        depth = 0
        balanced = True
        for ch in inner:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth < 0:
                    balanced = False
                    break
        if balanced and depth == 0:
            clean = inner.strip()
        else:
            break

    # Try to make it readable
    rule_text = f"In {table}, {clean}"

    return RuleMapping(
        table_name=table,
        constraint_name=constraint.constraint_name,
        rule_text=rule_text,
        category="constraint",
    )


# ---------------------------------------------------------------------------
# Schema heuristics: find name/description columns, detect catalog tables
# ---------------------------------------------------------------------------


_NAME_CANDIDATES = ["name", "title", "label", "display_name", "slug"]
_DESCRIPTION_CANDIDATES = [
    "description",
    "notes",
    "body",
    "content",
    "summary",
    "text",
    "markdown_content",
]


def find_name_column(schema: TableSchemaFull) -> str | None:
    """Find the best column to use as a concept name."""
    col_names = {c["name"].lower(): c["name"] for c in schema.columns}
    for candidate in _NAME_CANDIDATES:
        if candidate in col_names:
            return col_names[candidate]
    return None


def find_description_columns(schema: TableSchemaFull) -> list[str]:
    """Find columns suitable for concept description."""
    col_names = {c["name"].lower(): c["name"] for c in schema.columns}
    found = []
    for candidate in _DESCRIPTION_CANDIDATES:
        if candidate in col_names:
            found.append(col_names[candidate])
    return found


def auto_domain(schema: TableSchemaFull, database_name: str = "") -> str:
    """Suggest a domain name for a table.

    Heuristic: use schema_name if not 'public', else database_name,
    else table prefix (e.g. habit_templates → habit).
    """
    if schema.schema_name and schema.schema_name != "public":
        return schema.schema_name

    if database_name:
        return database_name

    # Try to extract prefix from table name
    parts = schema.name.split("_")
    if len(parts) >= 2:
        return parts[0]

    return schema.name


def detect_catalog_table(schema: TableSchemaFull) -> bool:
    """Detect if a table is a catalog/reference table (few writes, many reads).

    Heuristics:
    - Has a 'slug' column (implies human-readable stable identifier)
    - Has 'is_active' or 'is_public' flags
    - Name ends with '_templates' or '_types' or '_categories'
    - Has few FK columns but many non-FK columns (rich entity, not junction)
    """
    col_names = {c["name"].lower() for c in schema.columns}
    table_lower = schema.name.lower()

    # Slug → catalog
    if "slug" in col_names:
        return True

    # Template/type/category suffix (but NOT junction tables like meal_food_items)
    _CATALOG_SUFFIXES = ("_templates", "_types", "_categories")
    if any(table_lower.endswith(s) for s in _CATALOG_SUFFIXES):
        return True

    # _items suffix: only catalog if NOT a junction table (has ≤1 FK)
    if table_lower.endswith("_items"):
        fk_count = len(schema.foreign_keys) if hasattr(schema, "foreign_keys") else 0
        if fk_count <= 1:
            return True

    # is_active/is_public flags
    if "is_active" in col_names or "is_public" in col_names:
        return True

    return False


# ---------------------------------------------------------------------------
# Cross-database edge detection
# ---------------------------------------------------------------------------


def detect_cross_db_edges_by_naming(schemas: list[SchemaGraph]) -> list[CrossDatabaseEdge]:
    """Detect cross-database edges by column naming conventions.

    Looks for columns like 'exercise_id' in one database that reference
    a table called 'exercises' or 'movements' in another database.
    Also catches pattern where column name matches table name in another DB.
    """
    # Build lookup: table_name → (database_name, table)
    table_index: dict[str, list[tuple[str, TableSchemaFull]]] = {}
    for sg in schemas:
        for t in sg.tables:
            table_index.setdefault(t.name, []).append((sg.database_name, t))
            # Also index by singular form
            singular = t.name.rstrip("s") if t.name.endswith("s") else t.name
            if singular != t.name:
                table_index.setdefault(singular, []).append((sg.database_name, t))

    edges: list[CrossDatabaseEdge] = []

    for sg in schemas:
        for table in sg.tables:
            for col in table.columns:
                col_name = col["name"].lower()
                if not col_name.endswith("_id"):
                    continue
                # Already an FK within the same database? Skip.
                is_local_fk = any(
                    fk.source_column == col["name"] for fk in table.foreign_keys
                )
                if is_local_fk:
                    continue

                # Extract referenced entity: exercise_id → exercise
                ref_entity = col_name[:-3]  # Strip "_id"

                # Look for matching table in OTHER databases
                candidates = table_index.get(ref_entity, []) + table_index.get(ref_entity + "s", [])
                for target_db, target_table in candidates:
                    if target_db == sg.database_name:
                        continue  # Same database, not cross-DB

                    pk_cols = target_table.pk_columns
                    target_col = pk_cols[0] if pk_cols else "id"

                    edges.append(
                        CrossDatabaseEdge(
                            source_database=sg.database_name,
                            source_table=table.name,
                            source_column=col["name"],
                            target_database=target_db,
                            target_table=target_table.name,
                            target_column=target_col,
                            relation_type=RelationType.USES.value,
                            confidence=0.7,
                            description=(
                                f"{table.name}.{col['name']} → "
                                f"{target_table.name}.{target_col} (naming convention)"
                            ),
                        )
                    )

    return edges


def create_user_identity_edges(schemas: list[SchemaGraph]) -> list[CrossDatabaseEdge]:
    """Create edges unifying user_id columns across databases.

    When multiple databases reference user_id (UUID or string), this creates
    BELONGS_TO edges pointing to the canonical users table.
    """
    # Find the users table
    users_db = None
    users_table = None
    for sg in schemas:
        for t in sg.tables:
            if t.name == "users":
                users_db = sg.database_name
                users_table = t
                break
        if users_table:
            break

    if users_table is None:
        return []

    pk_col = users_table.pk_columns[0] if users_table.pk_columns else "id"

    edges: list[CrossDatabaseEdge] = []
    for sg in schemas:
        if sg.database_name == users_db:
            continue
        for table in sg.tables:
            for col in table.columns:
                if col["name"].lower() == "user_id":
                    edges.append(
                        CrossDatabaseEdge(
                            source_database=sg.database_name,
                            source_table=table.name,
                            source_column=col["name"],
                            target_database=users_db,
                            target_table="users",
                            target_column=pk_col,
                            relation_type=RelationType.BELONGS_TO.value,
                            confidence=0.9,
                            description=(
                                f"{table.name}.user_id → users.{pk_col} (identity unification)"
                            ),
                        )
                    )

    return edges
