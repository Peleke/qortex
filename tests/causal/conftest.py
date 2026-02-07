"""Shared fixtures for causal tests.

Canonical DAGs: chain, fork, collider, sprinkler, smoking.
Backend fixture with populated KG edges.
"""

from __future__ import annotations

import pytest

from qortex.causal.dag import CausalDAG
from qortex.causal.types import CausalDirection, CausalEdge
from qortex.core.memory import InMemoryBackend
from qortex.core.models import ConceptEdge, ConceptNode, RelationType

# =============================================================================
# Helper
# =============================================================================


def _edge(src: str, tgt: str, rel: str = "requires", strength: float = 1.0) -> CausalEdge:
    return CausalEdge(
        source_id=src,
        target_id=tgt,
        relation_type=rel,
        direction=CausalDirection.FORWARD,
        strength=strength,
    )


# =============================================================================
# DAG fixtures
# =============================================================================


@pytest.fixture
def chain_dag() -> CausalDAG:
    """A → B → C"""
    return CausalDAG.from_edges(
        [_edge("A", "B"), _edge("B", "C")],
        node_names={"A": "A", "B": "B", "C": "C"},
    )


@pytest.fixture
def fork_dag() -> CausalDAG:
    """B → A, B → C (B is a common cause)"""
    return CausalDAG.from_edges(
        [_edge("B", "A"), _edge("B", "C")],
        node_names={"A": "A", "B": "B", "C": "C"},
    )


@pytest.fixture
def collider_dag() -> CausalDAG:
    """A → B, C → B (B is a collider)"""
    return CausalDAG.from_edges(
        [_edge("A", "B"), _edge("C", "B")],
        node_names={"A": "A", "B": "B", "C": "C"},
    )


@pytest.fixture
def sprinkler_dag() -> CausalDAG:
    """Season → Rain, Season → Sprinkler, Rain → Wet, Sprinkler → Wet"""
    return CausalDAG.from_edges(
        [
            _edge("Season", "Rain"),
            _edge("Season", "Sprinkler"),
            _edge("Rain", "Wet"),
            _edge("Sprinkler", "Wet"),
        ],
        node_names={
            "Season": "Season",
            "Rain": "Rain",
            "Sprinkler": "Sprinkler",
            "Wet": "Wet",
        },
    )


@pytest.fixture
def smoking_dag() -> CausalDAG:
    """Smoking → Tar → Cancer, Smoking → Cancer"""
    return CausalDAG.from_edges(
        [
            _edge("Smoking", "Tar"),
            _edge("Tar", "Cancer"),
            _edge("Smoking", "Cancer"),
        ],
        node_names={
            "Smoking": "Smoking",
            "Tar": "Tar",
            "Cancer": "Cancer",
        },
    )


# =============================================================================
# Backend fixture
# =============================================================================


@pytest.fixture
def backend_with_graph() -> InMemoryBackend:
    """InMemoryBackend populated with a small concept graph.

    Domain "test":
        A --requires--> B --uses--> C --requires--> D
    """
    backend = InMemoryBackend()
    backend.connect()
    backend.create_domain("test")

    for nid in ["A", "B", "C", "D"]:
        backend.add_node(
            ConceptNode(
                id=nid,
                name=nid,
                description=f"Concept {nid}",
                domain="test",
                source_id="test_source",
            )
        )

    backend.add_edge(ConceptEdge(source_id="A", target_id="B", relation_type=RelationType.REQUIRES))
    backend.add_edge(ConceptEdge(source_id="B", target_id="C", relation_type=RelationType.USES))
    backend.add_edge(ConceptEdge(source_id="C", target_id="D", relation_type=RelationType.REQUIRES))

    return backend
