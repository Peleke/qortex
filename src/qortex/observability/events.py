"""Typed event dataclasses for qortex observability.

All events are frozen (immutable) dataclasses. Modules emit these —
they don't know about metrics, traces, or logs. Subscribers handle routing.

Grouped by domain: query lifecycle, PPR convergence, teleportation factors,
edge promotion, retrieval, interoception lifecycle, enrichment, ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Query Lifecycle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryStarted:
    query_id: str
    query_text: str
    domains: tuple[str, ...] | None
    mode: str  # "vec" | "graph"
    top_k: int
    timestamp: str  # ISO


@dataclass(frozen=True)
class QueryCompleted:
    query_id: str
    latency_ms: float
    seed_count: int
    result_count: int
    activated_nodes: int
    mode: str
    timestamp: str


@dataclass(frozen=True)
class QueryFailed:
    query_id: str
    error: str
    stage: str  # "embedding" | "vec_search" | "ppr" | "scoring"
    timestamp: str


# ---------------------------------------------------------------------------
# PPR Convergence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PPRStarted:
    query_id: str | None
    node_count: int
    seed_count: int
    damping_factor: float
    extra_edge_count: int


@dataclass(frozen=True)
class PPRConverged:
    query_id: str | None
    iterations: int
    final_diff: float
    node_count: int
    nonzero_scores: int
    latency_ms: float


@dataclass(frozen=True)
class PPRDiverged:
    query_id: str | None
    iterations: int
    final_diff: float
    node_count: int


# ---------------------------------------------------------------------------
# Teleportation Factors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorUpdated:
    node_id: str
    query_id: str
    outcome: str  # "accepted" | "rejected" | "partial"
    old_factor: float
    new_factor: float
    delta: float
    clamped: bool


@dataclass(frozen=True)
class FactorsPersisted:
    path: str
    count: int
    timestamp: str


@dataclass(frozen=True)
class FactorsLoaded:
    path: str
    count: int
    timestamp: str


@dataclass(frozen=True)
class FactorDriftSnapshot:
    """Emitted periodically — tracks whether factors are converging or diverging."""

    count: int
    mean: float
    min_val: float
    max_val: float
    boosted: int
    penalized: int
    entropy: float  # Shannon entropy of factor distribution


# ---------------------------------------------------------------------------
# Edge Promotion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnlineEdgeRecorded:
    source_id: str
    target_id: str
    score: float
    hit_count: int
    buffer_size: int


@dataclass(frozen=True)
class EdgePromoted:
    source_id: str
    target_id: str
    hit_count: int
    avg_score: float


@dataclass(frozen=True)
class BufferFlushed:
    promoted: int
    remaining: int
    total_promoted_lifetime: int
    kg_coverage: float | None
    timestamp: str


# ---------------------------------------------------------------------------
# Retrieval Adapter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VecSearchCompleted:
    query_id: str
    candidates: int
    fetch_k: int
    latency_ms: float


@dataclass(frozen=True)
class OnlineEdgesGenerated:
    query_id: str
    edge_count: int
    threshold: float
    seed_count: int


@dataclass(frozen=True)
class KGCoverageComputed:
    query_id: str
    persistent_edges: int
    online_edges: int
    coverage: float  # persistent / total


@dataclass(frozen=True)
class FeedbackReceived:
    query_id: str
    outcomes: int
    accepted: int
    rejected: int
    partial: int
    source: str  # "mcp" | "openclaw" | "buildlog"


# ---------------------------------------------------------------------------
# Interoception Lifecycle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InteroceptionStarted:
    factors_loaded: int
    buffer_loaded: int
    teleportation_enabled: bool


@dataclass(frozen=True)
class InteroceptionShutdown:
    factors_persisted: int
    buffer_persisted: int
    summary: dict  # from .summary()


# ---------------------------------------------------------------------------
# Enrichment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnrichmentCompleted:
    rule_count: int
    succeeded: int
    failed: int
    backend_type: str
    latency_ms: float


@dataclass(frozen=True)
class EnrichmentFallback:
    reason: str
    rule_count: int


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ManifestIngested:
    domain: str
    node_count: int
    edge_count: int
    rule_count: int
    source_id: str
    latency_ms: float
