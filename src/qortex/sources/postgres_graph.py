"""PostgreSQL graph ingestor: schema → knowledge graph via asyncpg.

Implements the GraphIngestor protocol for PostgreSQL databases.
Discovers schemas with full constraint metadata, maps them to graph
structure using mechanical rules, and ingests into qortex backends.

Requires: asyncpg (via qortex[source-postgres])
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qortex.core.models import ConceptEdge, ConceptNode, ExplicitRule, RelationType
from qortex.sources.graph_ingestor import (
    CheckConstraintInfo,
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
    detect_catalog_table,
    find_description_columns,
    find_name_column,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pg_constraint ON DELETE code → human-readable
# ---------------------------------------------------------------------------

_DELETE_CODE_MAP = {
    "a": "NO ACTION",
    "r": "RESTRICT",
    "c": "CASCADE",
    "n": "SET NULL",
    "d": "SET DEFAULT",
}


class PostgresGraphIngestor:
    """Async PostgreSQL → knowledge graph ingestor.

    discover_schema() → SchemaGraph (async, queries information_schema + pg_constraint)
    map_schema() → GraphMapping (sync, pure logic via mapping_rules)
    ingest() → dict[str, int] (async, creates ConceptNodes/Edges/Rules in backend)
    """

    def __init__(
        self,
        config: Any = None,
        ingest_config: Any = None,
        backend: Any = None,
        embedding_model: Any = None,
    ) -> None:
        self._config = config
        self._ingest_config = ingest_config
        self._backend = backend
        self._embedding_model = embedding_model
        self._conn: Any = None

    # ------------------------------------------------------------------
    # discover_schema: async database introspection
    # ------------------------------------------------------------------

    async def discover_schema(
        self,
        conn: Any | None = None,
        database_name: str = "",
        source_id: str = "",
    ) -> SchemaGraph:
        """Discover full schema with FK, CHECK, UNIQUE constraints.

        Args:
            conn: asyncpg connection (uses self._conn if None).
            database_name: Name for the SchemaGraph.
            source_id: Source identifier.
        """
        conn = conn or self._conn
        if conn is None:
            raise RuntimeError("No connection. Pass conn or set self._conn.")

        schemas_list = ["public"]
        if self._config is not None:
            schemas_list = getattr(self._config, "schemas", ["public"]) or ["public"]
            source_id = source_id or getattr(self._config, "source_id", "")
            database_name = database_name or source_id

        tables: list[TableSchemaFull] = []

        for schema_name in schemas_list:
            table_rows = await conn.fetch(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = $1 AND table_type = 'BASE TABLE'
                ORDER BY table_name
                """,
                schema_name,
            )

            for trow in table_rows:
                table_name = trow["table_name"]

                # Columns
                col_rows = await conn.fetch(
                    """
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = $1 AND table_name = $2
                    ORDER BY ordinal_position
                    """,
                    schema_name,
                    table_name,
                )

                # PKs
                pk_rows = await conn.fetch(
                    """
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    WHERE tc.table_schema = $1
                        AND tc.table_name = $2
                        AND tc.constraint_type = 'PRIMARY KEY'
                    """,
                    schema_name,
                    table_name,
                )
                pk_cols = {r["column_name"] for r in pk_rows}

                columns = [
                    {
                        "name": crow["column_name"],
                        "data_type": crow["data_type"],
                        "nullable": crow["is_nullable"] == "YES",
                        "is_pk": crow["column_name"] in pk_cols,
                        "default": crow["column_default"],
                    }
                    for crow in col_rows
                ]

                # FKs via pg_constraint (includes ON DELETE action)
                fk_rows = await conn.fetch(
                    """
                    SELECT
                        con.conname AS constraint_name,
                        att_src.attname AS source_column,
                        cls_tgt.relname AS target_table,
                        att_tgt.attname AS target_column,
                        con.confdeltype AS on_delete_code
                    FROM pg_constraint con
                    JOIN pg_class cls_src ON con.conrelid = cls_src.oid
                    JOIN pg_namespace nsp ON cls_src.relnamespace = nsp.oid
                    JOIN pg_class cls_tgt ON con.confrelid = cls_tgt.oid
                    JOIN pg_attribute att_src
                        ON att_src.attrelid = cls_src.oid
                        AND att_src.attnum = ANY(con.conkey)
                    JOIN pg_attribute att_tgt
                        ON att_tgt.attrelid = cls_tgt.oid
                        AND att_tgt.attnum = ANY(con.confkey)
                    WHERE con.contype = 'f'
                        AND nsp.nspname = $1
                        AND cls_src.relname = $2
                    """,
                    schema_name,
                    table_name,
                )

                foreign_keys = [
                    ForeignKeyInfo(
                        constraint_name=r["constraint_name"],
                        source_table=table_name,
                        source_column=r["source_column"],
                        target_table=r["target_table"],
                        target_column=r["target_column"],
                        on_delete=_DELETE_CODE_MAP.get(r["on_delete_code"], "NO ACTION"),
                    )
                    for r in fk_rows
                ]

                # CHECK constraints
                check_rows = await conn.fetch(
                    """
                    SELECT con.conname AS constraint_name,
                           pg_get_constraintdef(con.oid) AS check_clause
                    FROM pg_constraint con
                    JOIN pg_class cls ON con.conrelid = cls.oid
                    JOIN pg_namespace nsp ON cls.relnamespace = nsp.oid
                    WHERE con.contype = 'c'
                        AND nsp.nspname = $1
                        AND cls.relname = $2
                    """,
                    schema_name,
                    table_name,
                )

                check_constraints = [
                    CheckConstraintInfo(
                        constraint_name=r["constraint_name"],
                        table_name=table_name,
                        check_clause=r["check_clause"],
                    )
                    for r in check_rows
                ]

                # UNIQUE constraints
                unique_rows = await conn.fetch(
                    """
                    SELECT
                        con.conname AS constraint_name,
                        array_agg(att.attname ORDER BY att.attnum) AS columns
                    FROM pg_constraint con
                    JOIN pg_class cls ON con.conrelid = cls.oid
                    JOIN pg_namespace nsp ON cls.relnamespace = nsp.oid
                    JOIN pg_attribute att
                        ON att.attrelid = cls.oid
                        AND att.attnum = ANY(con.conkey)
                    WHERE con.contype = 'u'
                        AND nsp.nspname = $1
                        AND cls.relname = $2
                    GROUP BY con.conname
                    """,
                    schema_name,
                    table_name,
                )

                unique_constraints = [
                    UniqueConstraintInfo(
                        constraint_name=r["constraint_name"],
                        table_name=table_name,
                        columns=list(r["columns"]),
                    )
                    for r in unique_rows
                ]

                # Row count
                count = await conn.fetchval(
                    f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}"'  # noqa: S608
                )

                tables.append(
                    TableSchemaFull(
                        name=table_name,
                        schema_name=schema_name,
                        columns=columns,
                        row_count=count or 0,
                        foreign_keys=foreign_keys,
                        check_constraints=check_constraints,
                        unique_constraints=unique_constraints,
                    )
                )

        return SchemaGraph(
            source_id=source_id or database_name,
            database_name=database_name,
            tables=tables,
        )

    # ------------------------------------------------------------------
    # map_schema: pure logic (no I/O)
    # ------------------------------------------------------------------

    def map_schema(
        self,
        schema: SchemaGraph,
        domain_map: dict[str, str] | None = None,
    ) -> GraphMapping:
        """Map database schema to graph structure.

        Uses mapping_rules for FK classification, constraint→rule,
        catalog detection, and name/description column heuristics.
        """
        import fnmatch

        table_mappings: list[TableMapping] = []
        edge_mappings: list[EdgeMapping] = []
        rule_mappings: list[RuleMapping] = []

        domain_map = domain_map or {}
        if self._config is not None:
            domain_map = domain_map or getattr(self._config, "domain_map", {})

        for table in schema.tables:
            # Resolve domain
            domain = None
            if domain_map:
                domain = next(
                    (d for pat, d in domain_map.items() if fnmatch.fnmatch(table.name, pat)),
                    None,
                )
            if domain is None:
                domain = auto_domain(table, schema.database_name)

            name_col = find_name_column(table)
            desc_cols = find_description_columns(table)
            is_catalog = detect_catalog_table(table)

            table_mappings.append(
                TableMapping(
                    table_name=table.name,
                    domain=domain,
                    name_column=name_col,
                    description_columns=desc_cols,
                    is_catalog=is_catalog,
                )
            )

            # FK → edge mappings
            for fk in table.foreign_keys:
                rel_type = classify_fk_relation(fk, table)
                edge_mappings.append(
                    EdgeMapping(
                        source_table=fk.source_table,
                        target_table=fk.target_table,
                        fk_column=fk.source_column,
                        relation_type=rel_type,
                    )
                )

            # CHECK → rule mappings
            for check in table.check_constraints:
                rule = constraint_to_rule(check, table.name)
                rule_mappings.append(rule)

        return GraphMapping(
            tables=table_mappings,
            edges=edge_mappings,
            rules=rule_mappings,
        )

    # ------------------------------------------------------------------
    # ingest: write ConceptNodes/Edges/Rules to backend
    # ------------------------------------------------------------------

    async def ingest(
        self,
        mapping: GraphMapping,
        schema: SchemaGraph,
        conn: Any | None = None,
        backend: Any | None = None,
        embedding_model: Any | None = None,
    ) -> dict[str, int]:
        """Ingest mapped data into qortex backend.

        Creates ConceptNodes for catalog table rows, lightweight nodes
        for non-catalog rows, edges from FKs, rules from CHECK constraints.

        Returns counts: {concepts, edges, rules}.
        """
        conn = conn or self._conn
        backend = backend or self._backend
        embedding_model = embedding_model or self._embedding_model

        if conn is None:
            raise RuntimeError("No connection.")
        if backend is None:
            raise RuntimeError("No backend.")

        counts = {"concepts": 0, "edges": 0, "rules": 0}
        table_map = {tm.table_name: tm for tm in mapping.tables}

        # 1. Ensure domains exist
        for tm in mapping.tables:
            if backend.get_domain(tm.domain) is None:
                backend.create_domain(tm.domain)

        # 2. Create nodes — track pk→node_id for edge wiring
        embed_catalogs = True
        if self._ingest_config is not None:
            embed_catalogs = getattr(self._ingest_config, "embed_catalog_tables", True)

        # table_name → { pk_value → node_id }
        node_id_map: dict[str, dict[str, str]] = {}

        for table_schema in schema.tables:
            tm = table_map.get(table_schema.name)
            if tm is None:
                continue

            node_id_map[table_schema.name] = {}

            rows = await conn.fetch(
                f'SELECT * FROM "{table_schema.schema_name}"."{table_schema.name}"'  # noqa: S608
            )

            texts_for_embed: list[str] = []
            ids_for_embed: list[str] = []

            for row in rows:
                rd = dict(row)
                pk_cols = table_schema.pk_columns
                pk_val = (
                    ":".join(str(rd.get(pk, "")) for pk in pk_cols)
                    if pk_cols
                    else str(uuid.uuid4())
                )
                node_id = f"{schema.source_id}:{table_schema.name}:{pk_val}"
                node_id_map[table_schema.name][pk_val] = node_id

                # Build name + description
                name = str(rd.get(tm.name_column, pk_val)) if tm.name_column else pk_val
                desc_parts = [
                    str(rd.get(c, "")) for c in tm.description_columns if rd.get(c)
                ]
                description = ". ".join(desc_parts) if desc_parts else name

                props: dict[str, Any] = {"table": table_schema.name, "pk": pk_val}
                if tm.is_catalog:
                    props["is_catalog"] = True

                node = ConceptNode(
                    id=node_id,
                    name=name,
                    description=description,
                    domain=tm.domain,
                    source_id=schema.source_id,
                    properties=props,
                )
                backend.add_node(node)
                counts["concepts"] += 1

                # Queue catalog rows for embedding
                if tm.is_catalog and embed_catalogs and embedding_model is not None:
                    texts_for_embed.append(f"{name}: {description}")
                    ids_for_embed.append(node_id)

            # Batch embed catalog rows
            if texts_for_embed and embedding_model is not None:
                embeddings = embedding_model.embed(texts_for_embed)
                for nid, emb in zip(ids_for_embed, embeddings):
                    backend.add_embedding(nid, emb)

        # 3. Create edges from FK mappings
        for edge_m in mapping.edges:
            source_schema = schema.get_table(edge_m.source_table)
            if source_schema is None:
                continue

            source_nodes = node_id_map.get(edge_m.source_table, {})
            target_nodes = node_id_map.get(edge_m.target_table, {})
            if not source_nodes or not target_nodes:
                continue

            rows = await conn.fetch(
                f'SELECT * FROM "{source_schema.schema_name}"."{source_schema.name}"'  # noqa: S608
            )

            rel_type = RelationType(edge_m.relation_type)

            for row in rows:
                rd = dict(row)
                fk_value = rd.get(edge_m.fk_column)
                if fk_value is None:
                    continue

                pk_cols = source_schema.pk_columns
                src_pk = (
                    ":".join(str(rd.get(pk, "")) for pk in pk_cols) if pk_cols else ""
                )
                src_node_id = source_nodes.get(src_pk)
                if src_node_id is None:
                    continue

                tgt_node_id = target_nodes.get(str(fk_value))
                if tgt_node_id is None:
                    continue

                edge = ConceptEdge(
                    source_id=src_node_id,
                    target_id=tgt_node_id,
                    relation_type=rel_type,
                    confidence=edge_m.confidence,
                    properties={"fk_column": edge_m.fk_column},
                )
                backend.add_edge(edge)
                counts["edges"] += 1

        # 4. Create rules from CHECK constraints
        extract_rules = True
        if self._ingest_config is not None:
            extract_rules = getattr(self._ingest_config, "extract_rules", True)

        if extract_rules:
            for rule_m in mapping.rules:
                domain = (
                    table_map[rule_m.table_name].domain
                    if rule_m.table_name in table_map
                    else "default"
                )
                rule = ExplicitRule(
                    id=f"{schema.source_id}:rule:{rule_m.constraint_name}",
                    text=rule_m.rule_text,
                    domain=domain,
                    category=rule_m.category,
                    source_id=schema.source_id,
                )
                backend.add_rule(rule)
                counts["rules"] += 1

        return counts

    async def run(
        self,
        conn: Any | None = None,
        backend: Any | None = None,
        embedding_model: Any | None = None,
    ) -> dict[str, int]:
        """Convenience: discover → map → ingest in one call."""
        conn = conn or self._conn
        backend = backend or self._backend
        embedding_model = embedding_model or self._embedding_model

        source_id = getattr(self._config, "source_id", "") if self._config else ""
        schema = await self.discover_schema(conn, source_id=source_id)
        mapping = self.map_schema(schema)
        return await self.ingest(mapping, schema, conn, backend, embedding_model)
