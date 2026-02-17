"""Declarative metric definitions: single source of truth for all qortex metrics.

Every metric is defined once here. The factory creates OTel instruments from
these definitions. The handlers wire events to instruments. No metric is
defined anywhere else.

Adding a metric:
    1. Add a MetricDef to METRICS below
    2. Add a handler in metrics_handlers.py
    3. Done. Both OTLP push and Prometheus /metrics get it automatically.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MetricType(Enum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"


@dataclass(frozen=True)
class MetricDef:
    """A single metric definition."""

    name: str
    type: MetricType
    description: str
    labels: tuple[str, ...] = ()
    buckets: tuple[float, ...] | None = None  # histograms only


# All 48 metrics in one registry.
# Names follow Prometheus conventions:
#   Counter:   "qortex_foo"          → exported as "qortex_foo_total"
#   Histogram: "qortex_bar_seconds"  → exported as "qortex_bar_seconds_bucket" etc.
#   Gauge:     "qortex_baz"          → exported as "qortex_baz"
METRICS: tuple[MetricDef, ...] = (
    # ── Query lifecycle ──────────────────────────────────────────────
    MetricDef(
        "qortex_queries", MetricType.COUNTER,
        "Total queries", ("mode",),
    ),
    MetricDef(
        "qortex_query_duration_seconds", MetricType.HISTOGRAM,
        "Query latency",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    ),
    MetricDef(
        "qortex_query_errors", MetricType.COUNTER,
        "Query errors", ("stage",),
    ),
    MetricDef(
        "qortex_query_overhead_seconds", MetricType.HISTOGRAM,
        "Query overhead (our code, excluding external calls)",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
    ),
    # ── PPR ──────────────────────────────────────────────────────────
    MetricDef(
        "qortex_ppr_started", MetricType.COUNTER,
        "PPR executions started",
    ),
    MetricDef(
        "qortex_ppr_iterations", MetricType.HISTOGRAM,
        "PPR iterations to converge",
        buckets=(1, 5, 10, 20, 50, 100),
    ),
    # ── Factors ──────────────────────────────────────────────────────
    MetricDef(
        "qortex_factor_updates", MetricType.COUNTER,
        "Factor update events", ("outcome",),
    ),
    MetricDef(
        "qortex_factor_mean", MetricType.GAUGE,
        "Mean teleportation factor",
    ),
    MetricDef(
        "qortex_factor_entropy", MetricType.GAUGE,
        "Factor distribution entropy",
    ),
    MetricDef(
        "qortex_factors_active", MetricType.GAUGE,
        "Active teleportation factors",
    ),
    # ── Graph node/edge counts ────────────────────────────────────────
    MetricDef(
        "qortex_graph_nodes_created", MetricType.COUNTER,
        "Graph nodes created", ("domain", "origin"),
    ),
    MetricDef(
        "qortex_graph_edges_created", MetricType.COUNTER,
        "Graph edges created", ("domain", "origin"),
    ),
    MetricDef(
        "qortex_graph_node_count", MetricType.GAUGE,
        "Total nodes in graph",
    ),
    MetricDef(
        "qortex_graph_edge_count", MetricType.GAUGE,
        "Total edges in graph",
    ),
    # ── Edges ────────────────────────────────────────────────────────
    MetricDef(
        "qortex_buffer_edges", MetricType.GAUGE,
        "Buffered online edges",
    ),
    MetricDef(
        "qortex_edges_promoted", MetricType.COUNTER,
        "Lifetime edge promotions",
    ),
    MetricDef(
        "qortex_kg_coverage", MetricType.GAUGE,
        "KG coverage ratio",
    ),
    MetricDef(
        "qortex_online_edges_generated", MetricType.COUNTER,
        "Online edge generation events",
    ),
    MetricDef(
        "qortex_online_edge_count", MetricType.GAUGE,
        "Online edges in last query",
    ),
    # ── Feedback ─────────────────────────────────────────────────────
    MetricDef(
        "qortex_feedback", MetricType.COUNTER,
        "Feedback events", ("outcome",),
    ),
    # ── Vec search ───────────────────────────────────────────────────
    MetricDef(
        "qortex_vec_search_duration_seconds", MetricType.HISTOGRAM,
        "Vec search latency",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
    ),
    MetricDef(
        "qortex_vec_search_candidates", MetricType.HISTOGRAM,
        "Candidates per search",
        buckets=(1, 5, 10, 20, 50, 100),
    ),
    MetricDef(
        "qortex_vec_search_top_score", MetricType.GAUGE,
        "Top cosine sim score",
    ),
    MetricDef(
        "qortex_vec_search_score_spread", MetricType.GAUGE,
        "Score spread",
    ),
    MetricDef(
        "qortex_vec_seed_yield", MetricType.GAUGE,
        "Seed yield ratio",
    ),
    # ── Vec index ────────────────────────────────────────────────────
    MetricDef(
        "qortex_vec_add", MetricType.COUNTER,
        "Vec index add ops", ("index_type",),
    ),
    MetricDef(
        "qortex_vec_add_duration_seconds", MetricType.HISTOGRAM,
        "Vec add latency",
        buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5),
    ),
    MetricDef(
        "qortex_vec_index_size", MetricType.GAUGE,
        "Vectors in index",
    ),
    # ── Enrichment ───────────────────────────────────────────────────
    MetricDef(
        "qortex_enrichment", MetricType.COUNTER,
        "Enrichment runs", ("backend_type",),
    ),
    MetricDef(
        "qortex_enrichment_duration_seconds", MetricType.HISTOGRAM,
        "Enrichment latency",
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
    ),
    MetricDef(
        "qortex_enrichment_fallbacks", MetricType.COUNTER,
        "Enrichment fallbacks",
    ),
    # ── Ingestion ────────────────────────────────────────────────────
    MetricDef(
        "qortex_manifests_ingested", MetricType.COUNTER,
        "Manifests ingested", ("domain",),
    ),
    MetricDef(
        "qortex_ingest_duration_seconds", MetricType.HISTOGRAM,
        "Ingest latency",
        buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
    ),
    # ── Online indexing ─────────────────────────────────────────────
    MetricDef(
        "qortex_messages_ingested", MetricType.COUNTER,
        "Session messages ingested", ("role",),
    ),
    MetricDef(
        "qortex_message_ingest_duration_seconds", MetricType.HISTOGRAM,
        "Message ingest latency",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
    ),
    MetricDef(
        "qortex_tool_results_ingested", MetricType.COUNTER,
        "Tool results ingested", ("tool_name",),
    ),
    MetricDef(
        "qortex_tool_result_ingest_duration_seconds", MetricType.HISTOGRAM,
        "Tool result ingest latency",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
    ),
    # ── Concept Extraction ──────────────────────────────────────────
    MetricDef(
        "qortex_concepts_extracted", MetricType.COUNTER,
        "Concepts extracted from chunks", ("strategy", "domain"),
    ),
    MetricDef(
        "qortex_relations_extracted", MetricType.COUNTER,
        "Relations extracted from chunks", ("strategy", "domain"),
    ),
    MetricDef(
        "qortex_extraction_duration_seconds", MetricType.HISTOGRAM,
        "Per-chunk extraction latency",
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    ),
    MetricDef(
        "qortex_extraction_pipeline_duration_seconds", MetricType.HISTOGRAM,
        "Full extraction pipeline latency (all chunks)",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    ),
    MetricDef(
        "qortex_extraction_concepts_per_chunk", MetricType.HISTOGRAM,
        "Concepts extracted per chunk",
        buckets=(0, 1, 2, 5, 10, 20, 50),
    ),
    MetricDef(
        "qortex_extraction_relations_per_chunk", MetricType.HISTOGRAM,
        "Relations extracted per chunk",
        buckets=(0, 1, 2, 5, 10, 20),
    ),
    MetricDef(
        "qortex_extractions", MetricType.COUNTER,
        "Extraction pipeline runs", ("strategy",),
    ),
    # ── Learning ─────────────────────────────────────────────────────
    MetricDef(
        "qortex_learning_selections", MetricType.COUNTER,
        "Learning selections", ("learner", "baseline"),
    ),
    MetricDef(
        "qortex_learning_observations", MetricType.COUNTER,
        "Learning observations", ("learner", "outcome"),
    ),
    MetricDef(
        "qortex_learning_posterior_mean", MetricType.GAUGE,
        "Posterior mean", ("learner", "arm_id"),
    ),
    MetricDef(
        "qortex_learning_arm_pulls", MetricType.COUNTER,
        "Arm pulls", ("learner", "arm_id"),
    ),
    MetricDef(
        "qortex_learning_token_budget_used", MetricType.HISTOGRAM,
        "Token budget used",
        buckets=(100, 500, 1000, 2000, 4000, 8000, 16000),
    ),
    MetricDef(
        "qortex_learning_arm_alpha", MetricType.GAUGE,
        "Arm alpha (success count)", ("learner", "arm_id"),
    ),
    MetricDef(
        "qortex_learning_arm_beta", MetricType.GAUGE,
        "Arm beta (failure count)", ("learner", "arm_id"),
    ),
    # ── Credit propagation ───────────────────────────────────────────
    MetricDef(
        "qortex_credit_propagations", MetricType.COUNTER,
        "Credit propagations", ("learner",),
    ),
    MetricDef(
        "qortex_credit_concepts_per_propagation", MetricType.HISTOGRAM,
        "Concepts per propagation",
        buckets=(1, 5, 10, 25, 50, 100),
    ),
    MetricDef(
        "qortex_credit_alpha_delta", MetricType.COUNTER,
        "Cumulative alpha delta",
    ),
    MetricDef(
        "qortex_credit_beta_delta", MetricType.COUNTER,
        "Cumulative beta delta",
    ),
    # ── Carbon accounting ───────────────────────────────────────────
    MetricDef(
        "qortex_carbon_co2_grams", MetricType.COUNTER,
        "Cumulative CO2 emissions (grams)", ("provider", "model"),
    ),
    MetricDef(
        "qortex_carbon_water_ml", MetricType.COUNTER,
        "Cumulative water usage (milliliters)", ("provider", "model"),
    ),
    MetricDef(
        "qortex_carbon_tokens", MetricType.COUNTER,
        "Cumulative tokens processed for carbon tracking", ("provider", "model"),
    ),
    MetricDef(
        "qortex_carbon_confidence", MetricType.GAUGE,
        "Latest emission factor confidence",
    ),
    # ── Benchmarks ──────────────────────────────────────────────────
    MetricDef(
        "qortex_bench_round_accuracy", MetricType.GAUGE,
        "Benchmark round accuracy", ("benchmark", "round"),
    ),
    MetricDef(
        "qortex_bench_round_feedback", MetricType.COUNTER,
        "Benchmark feedback signals", ("benchmark", "outcome"),
    ),
    MetricDef(
        "qortex_bench_improvement", MetricType.GAUGE,
        "Benchmark learning curve improvement", ("benchmark",),
    ),
)
