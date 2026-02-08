"""Row serializers: convert database rows to text for embedding.

Two strategies:
- NaturalLanguageSerializer: "A food_item named 'Chicken Breast' with 165 calories..."
- KeyValueSerializer: "name=Chicken Breast, calories=165, protein=31g"
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from qortex.sources.base import TableSchema

# Columns to skip during serialization (internal/noise)
_SKIP_PATTERNS = {
    "id",
    "uuid",
    "created_at",
    "updated_at",
    "deleted_at",
    "created_by",
    "updated_by",
}

_SKIP_SUFFIXES = ("_id", "_uuid", "_at", "_hash")

# Columns to prioritize (put first in output)
_PRIORITY_NAMES = {"name", "title", "label", "display_name", "slug"}
_DESCRIPTION_NAMES = {"description", "notes", "body", "content", "summary", "text"}


def _is_internal_column(col_name: str) -> bool:
    """Check if a column is internal/noise and should be skipped."""
    lower = col_name.lower()
    if lower in _SKIP_PATTERNS:
        return True
    return any(lower.endswith(s) for s in _SKIP_SUFFIXES)


def _humanize_table_name(table_name: str) -> str:
    """Convert table_name to readable form: food_items → food item."""
    # Remove trailing 's' for plural
    name = (
        table_name.rstrip("s")
        if table_name.endswith("s") and not table_name.endswith("ss")
        else table_name
    )
    # Replace underscores with spaces
    return name.replace("_", " ")


def _humanize_column_name(col_name: str) -> str:
    """Convert column_name to readable form: max_heart_rate → max heart rate."""
    return col_name.replace("_", " ")


def _format_value(value: Any) -> str:
    """Format a value for serialization."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        # Remove trailing zeros
        return f"{value:g}"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if isinstance(value, dict):
        # Flatten one level of nested dict
        parts = [f"{k}: {v}" for k, v in value.items() if v is not None]
        return "; ".join(parts)
    return str(value)


@runtime_checkable
class RowSerializer(Protocol):
    """Protocol for row serialization strategies."""

    def serialize(
        self, table_name: str, row: dict[str, Any], schema: TableSchema | None = None
    ) -> str:
        """Convert a database row to text for embedding."""
        ...


class NaturalLanguageSerializer:
    """Serialize rows as readable natural language sentences.

    Skips internal columns (id, timestamps, UUIDs).
    Prioritizes name/title → description/notes → other.
    """

    def __init__(self, skip_internal: bool = True, max_length: int = 2000) -> None:
        self.skip_internal = skip_internal
        self.max_length = max_length

    def serialize(
        self, table_name: str, row: dict[str, Any], schema: TableSchema | None = None
    ) -> str:
        """Serialize a row to a natural language sentence."""
        entity = _humanize_table_name(table_name)

        # Partition columns into priority, description, and other
        priority_parts: list[str] = []
        desc_parts: list[str] = []
        other_parts: list[str] = []

        for col_name, value in row.items():
            if value is None or value == "":
                continue
            if self.skip_internal and _is_internal_column(col_name):
                continue

            formatted = _format_value(value)
            if not formatted:
                continue

            lower = col_name.lower()
            if lower in _PRIORITY_NAMES:
                priority_parts.append(f"named '{formatted}'")
            elif lower in _DESCRIPTION_NAMES:
                desc_parts.append(formatted)
            else:
                readable_col = _humanize_column_name(col_name)
                other_parts.append(f"{readable_col}: {formatted}")

        # Build sentence
        parts = []
        if priority_parts:
            parts.append(f"A {entity} {' '.join(priority_parts)}")
        else:
            parts.append(f"A {entity}")

        if desc_parts:
            parts.append(". ".join(desc_parts))

        if other_parts:
            parts.append(f"with {', '.join(other_parts)}")

        result = " ".join(parts)
        if len(result) > self.max_length:
            result = result[: self.max_length - 3] + "..."
        return result


class KeyValueSerializer:
    """Serialize rows as key=value pairs.

    Simple and deterministic. Good for exact matching.
    """

    def __init__(self, skip_internal: bool = True, separator: str = ", ") -> None:
        self.skip_internal = skip_internal
        self.separator = separator

    def serialize(
        self, table_name: str, row: dict[str, Any], schema: TableSchema | None = None
    ) -> str:
        """Serialize a row to key=value pairs."""
        parts = [f"table={table_name}"]
        for col_name, value in row.items():
            if value is None:
                continue
            if self.skip_internal and _is_internal_column(col_name):
                continue
            parts.append(f"{col_name}={_format_value(value)}")
        return self.separator.join(parts)
