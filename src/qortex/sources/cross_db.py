"""Cross-database edge application.

Resolves CrossDatabaseEdge descriptors into actual ConceptEdges
in the qortex backend, bridging entities across separate databases.
"""

from __future__ import annotations

import logging
from typing import Any

from qortex.core.models import ConceptEdge, RelationType
from qortex.sources.graph_ingestor import CrossDatabaseEdge, SchemaGraph
from qortex.sources.mapping_rules import (
    create_user_identity_edges,
    detect_cross_db_edges_by_naming,
)

logger = logging.getLogger(__name__)


def discover_all_cross_db(schemas: list[SchemaGraph]) -> list[CrossDatabaseEdge]:
    """Discover all cross-database edges from multiple schema graphs.

    Combines naming-convention detection and user-identity unification.
    """
    edges: list[CrossDatabaseEdge] = []
    # Identity edges first — higher confidence, more specific relation type
    edges.extend(create_user_identity_edges(schemas))
    edges.extend(detect_cross_db_edges_by_naming(schemas))

    # Deduplicate by (source_db, source_table, source_col, target_db, target_table)
    seen: set[tuple[str, str, str, str, str]] = set()
    unique: list[CrossDatabaseEdge] = []
    for e in edges:
        key = (
            e.source_database,
            e.source_table,
            e.source_column,
            e.target_database,
            e.target_table,
        )
        if key not in seen:
            seen.add(key)
            unique.append(e)

    return unique


def apply_cross_db_edges(
    edges: list[CrossDatabaseEdge],
    backend: Any,
    node_id_maps: dict[str, dict[str, dict[str, str]]] | None = None,
) -> int:
    """Resolve cross-DB edges to ConceptEdges in the backend.

    Args:
        edges: Cross-database edge descriptors.
        backend: GraphBackend to write edges to.
        node_id_maps: Optional lookup: {db_name: {table_name: {pk_value: node_id}}}.
            If not provided, attempts to resolve via backend node search.

    Returns:
        Number of edges created.
    """
    created = 0

    for edge in edges:
        rel_type = RelationType(edge.relation_type)

        # Try to find source and target nodes
        if node_id_maps:
            source_nodes = node_id_maps.get(edge.source_database, {}).get(
                edge.source_table, {}
            )
            target_nodes = node_id_maps.get(edge.target_database, {}).get(
                edge.target_table, {}
            )

            if not source_nodes or not target_nodes:
                logger.debug(
                    "No nodes found for cross-DB edge: %s.%s → %s.%s",
                    edge.source_database,
                    edge.source_table,
                    edge.target_database,
                    edge.target_table,
                )
                continue

            # Create schema-level edge (table-to-table) for cross-DB relationships.
            # Row-level edges require FK column values which aren't available here —
            # they were only in the original rows. Schema-level edges capture the
            # relationship structure for graph traversal.
            src_schema_id = f"{edge.source_database}:{edge.source_table}:_schema"
            tgt_schema_id = f"{edge.target_database}:{edge.target_table}:_schema"

            # Create schema placeholder nodes if they don't exist
            if backend.get_node(src_schema_id) is None:
                from qortex.core.models import ConceptNode

                backend.add_node(
                    ConceptNode(
                        id=src_schema_id,
                        name=f"{edge.source_table} (schema)",
                        description=f"Schema node for {edge.source_database}.{edge.source_table}",
                        domain="schema",
                        source_id=edge.source_database,
                        properties={"is_schema_node": True},
                    )
                )
            if backend.get_node(tgt_schema_id) is None:
                from qortex.core.models import ConceptNode

                backend.add_node(
                    ConceptNode(
                        id=tgt_schema_id,
                        name=f"{edge.target_table} (schema)",
                        description=f"Schema node for {edge.target_database}.{edge.target_table}",
                        domain="schema",
                        source_id=edge.target_database,
                        properties={"is_schema_node": True},
                    )
                )

            concept_edge = ConceptEdge(
                source_id=src_schema_id,
                target_id=tgt_schema_id,
                relation_type=rel_type,
                confidence=edge.confidence,
                properties={
                    "cross_db": True,
                    "source_column": edge.source_column,
                    "target_column": edge.target_column,
                    "description": edge.description,
                },
            )
            backend.add_edge(concept_edge)
            created += 1
            continue

        # Fallback (no node_id_maps): try schema-level edge if schema nodes exist
        src_id = f"{edge.source_database}:{edge.source_table}:_schema"
        tgt_id = f"{edge.target_database}:{edge.target_table}:_schema"

        if backend.get_node(src_id) is None or backend.get_node(tgt_id) is None:
            logger.debug(
                "Schema nodes not found for cross-DB edge: %s → %s",
                src_id,
                tgt_id,
            )
            continue

        concept_edge = ConceptEdge(
            source_id=src_id,
            target_id=tgt_id,
            relation_type=rel_type,
            confidence=edge.confidence,
            properties={
                "cross_db": True,
                "source_column": edge.source_column,
                "target_column": edge.target_column,
                "description": edge.description,
            },
        )
        backend.add_edge(concept_edge)
        created += 1

    return created
