"""Retrieval eval: QortexSimilarityMap vs vanilla vector search.

Tests precision and recall on a task/insight relevance matrix,
then runs a feedback loop to show qortex improving over rounds.
No LLM calls. No API keys. Zero cost.

Usage:
    .venv/bin/python benchmarks/autogen_tcm/eval_retrieval.py
    .venv/bin/python benchmarks/autogen_tcm/eval_retrieval.py --rounds 5
    .venv/bin/python benchmarks/autogen_tcm/eval_retrieval.py --verbose
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from qortex.adapters.autogen_tcm import QortexSimilarityMap


# ---------------------------------------------------------------------------
# Eval data: tasks, insights, and ground-truth relevance
# ---------------------------------------------------------------------------

# 15 tasks spanning auth, data, and infra domains
TASKS = [
    "How should a mobile app handle OAuth2 authentication securely?",
    "Compare token formats for API authentication",
    "Implement enterprise SSO for corporate applications",
    "Secure machine-to-machine auth in microservices",
    "Design a rate limiting strategy for public APIs",
    "Set up distributed caching with Redis for session management",
    "Implement database connection pooling for PostgreSQL",
    "Design a retry strategy with exponential backoff for HTTP clients",
    "Configure TLS termination at the load balancer",
    "Implement structured logging for a Python web service",
    "Design a circuit breaker for downstream service calls",
    "Set up health checks and readiness probes for Kubernetes",
    "Implement row-level security in PostgreSQL",
    "Design a webhook delivery system with retry guarantees",
    "Configure CORS for a multi-tenant SaaS API",
]

# 20 insights (things the system has learned)
INSIGHTS = [
    "Always use PKCE flow for public OAuth2 clients like SPAs and mobile apps",
    "JWTs should have short expiry (5-15 min) paired with rotating refresh tokens",
    "OpenID Connect adds an identity layer on top of OAuth2 for user authentication",
    "mTLS provides strong machine-to-machine auth without shared secrets",
    "API keys are simple but offer no user delegation or fine-grained scopes",
    "Token bucket algorithm handles bursty traffic better than fixed window",
    "Redis SCAN is preferred over KEYS for production cache enumeration",
    "Connection pool sizing: start with (2 * CPU cores) + disk spindles",
    "Exponential backoff with jitter prevents thundering herd on retries",
    "TLS 1.3 eliminates round trips and removes insecure cipher suites",
    "Structured logging with correlation IDs enables distributed tracing",
    "Circuit breaker half-open state should use a small percentage of traffic",
    "Kubernetes readiness probes should check downstream dependencies",
    "Row-level security policies in Postgres run as the querying role",
    "Webhook delivery needs idempotency keys to handle duplicate retries",
    "CORS preflight caching via Access-Control-Max-Age reduces OPTIONS requests",
    "SAML uses XML assertions for federated SSO in enterprise environments",
    "Session cookies require SameSite=Strict and HttpOnly for security",
    "gRPC uses HTTP/2 multiplexing for efficient machine-to-machine comms",
    "Bearer tokens in headers are preferred over URL query parameters",
]

# Ground truth: RELEVANCE[task_idx][insight_idx]
# 2 = strongly relevant, 1 = somewhat relevant, 0 = not relevant
RELEVANCE = [
    # Task 0: mobile OAuth2
    [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    # Task 1: token formats
    [0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2],
    # Task 2: enterprise SSO
    [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0],
    # Task 3: machine-to-machine
    [0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 1],
    # Task 4: rate limiting
    [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Task 5: Redis caching
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    # Task 6: connection pooling
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Task 7: retry strategy
    [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    # Task 8: TLS termination
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Task 9: structured logging
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # Task 10: circuit breaker
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0],
    # Task 11: health checks
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0],
    # Task 12: row-level security
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
    # Task 13: webhook delivery
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
    # Task 14: CORS config
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0],
]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    s = set(retrieved[:k])
    return len(s & relevant) / len(s) if s else 0.0


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    s = set(retrieved[:k])
    return len(s & relevant) / len(relevant) if relevant else 1.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, idx in enumerate(retrieved[:k])
        if idx in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg else 0.0


@dataclass
class RoundMetrics:
    round_num: int
    precision: float
    recall: float
    ndcg: float
    feedback_given: int = 0
    accepted: int = 0
    rejected: int = 0
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Eval logic
# ---------------------------------------------------------------------------


def populate_memory(ssm: QortexSimilarityMap) -> None:
    """Add all task/insight pairs where relevance == 2 (strongly relevant)."""
    for ti, task in enumerate(TASKS):
        for ii, insight in enumerate(INSIGHTS):
            if RELEVANCE[ti][ii] == 2:
                # Store with task as input (retrieval key), insight as output
                ssm.add_input_output_pair(task, insight)


def eval_round(
    ssm: QortexSimilarityMap,
    top_k: int = 5,
    threshold: float = 1.7,
) -> tuple[RoundMetrics, list[dict[str, Any]]]:
    """Run one eval round: query each task, measure precision/recall.

    Returns metrics and per-query details (for feedback).
    """
    all_precisions = []
    all_recalls = []
    all_ndcgs = []
    query_details = []

    for ti, task in enumerate(TASKS):
        # Ground truth: any insight with relevance > 0
        relevant = {ii for ii in range(len(INSIGHTS)) if RELEVANCE[ti][ii] > 0}
        strongly_relevant = {ii for ii in range(len(INSIGHTS)) if RELEVANCE[ti][ii] == 2}

        results = ssm.get_related_string_pairs(task, n_results=top_k, threshold=threshold)
        query_result = ssm.last_result  # Full QueryResult with item IDs

        # Map results back to insight indices
        retrieved_indices = []
        for input_text, output_text, distance in results:
            for ii, insight in enumerate(INSIGHTS):
                if output_text == insight and ii not in retrieved_indices:
                    retrieved_indices.append(ii)
                    break

        # Build item_id -> insight_idx from the full query result
        item_to_insight: dict[str, int] = {}
        if query_result:
            for item in query_result.items:
                pid = item.metadata.get("pair_id") if item.metadata else None
                if pid and pid in ssm._pairs:
                    _, out_text = ssm._pairs[pid]
                    for ii, insight in enumerate(INSIGHTS):
                        if out_text == insight:
                            item_to_insight[item.id] = ii
                            break

        p = precision_at_k(retrieved_indices, relevant, top_k)
        r = recall_at_k(retrieved_indices, relevant, top_k)
        n = ndcg_at_k(retrieved_indices, relevant, top_k)

        all_precisions.append(p)
        all_recalls.append(r)
        all_ndcgs.append(n)

        # Track which retrievals were correct/incorrect for feedback
        correct = set(retrieved_indices) & relevant
        incorrect = set(retrieved_indices) - relevant

        query_details.append({
            "task_idx": ti,
            "retrieved": retrieved_indices,
            "relevant": relevant,
            "correct": correct,
            "incorrect": incorrect,
            "precision": p,
            "recall": r,
            "query_id": query_result.query_id if query_result else None,
            "item_to_insight": item_to_insight,
        })

    metrics = RoundMetrics(
        round_num=0,
        precision=statistics.mean(all_precisions),
        recall=statistics.mean(all_recalls),
        ndcg=statistics.mean(all_ndcgs),
    )
    return metrics, query_details


def feed_back(
    ssm: QortexSimilarityMap,
    query_details: list[dict[str, Any]],
) -> tuple[int, int]:
    """Send feedback for each query's results.

    Uses the real query_id and item_ids from the eval round.
    Correct retrievals -> "accepted", incorrect -> "rejected".
    Returns (accepted_count, rejected_count).
    """
    accepted = 0
    rejected = 0

    for detail in query_details:
        query_id = detail.get("query_id")
        item_to_insight = detail.get("item_to_insight", {})
        correct = detail["correct"]
        incorrect = detail["incorrect"]

        if not query_id or not item_to_insight:
            continue

        # Build outcomes using real item IDs
        outcomes = {}
        for item_id, insight_idx in item_to_insight.items():
            if insight_idx in correct:
                outcomes[item_id] = "accepted"
                accepted += 1
            elif insight_idx in incorrect:
                outcomes[item_id] = "rejected"
                rejected += 1

        if outcomes:
            try:
                ssm.client.feedback(
                    query_id=query_id,
                    outcomes=outcomes,
                    source="autogen_tcm_eval",
                )
            except Exception as e:
                logger.debug(f"Feedback failed: {e}")

    return accepted, rejected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

import logging

logger = logging.getLogger(__name__)


def run_eval(
    learning_rounds: int = 3,
    top_k: int = 5,
    threshold: float = 1.7,
    verbose: bool = False,
) -> list[RoundMetrics]:
    """Run the full retrieval eval with learning loop.

    Returns list of RoundMetrics, one per round.
    """
    import tempfile

    tmp = tempfile.mkdtemp(prefix="qortex_tcm_eval_")
    ssm = QortexSimilarityMap(reset=True, path_to_db_dir=tmp)

    print("Populating memory...")
    t0 = time.perf_counter()
    populate_memory(ssm)
    pair_count = len(ssm._pairs)
    print(f"  Stored {pair_count} task/insight pairs in {time.perf_counter()-t0:.1f}s")

    print("Building edges...")
    t0 = time.perf_counter()
    edge_count = ssm.build_edges(sim_threshold=0.3)
    print(f"  Created {edge_count} edges in {time.perf_counter()-t0:.1f}s")

    rounds: list[RoundMetrics] = []

    # Round 0: baseline (no feedback yet)
    print("\n--- Round 0 (baseline) ---")
    t0 = time.perf_counter()
    metrics, details = eval_round(ssm, top_k=top_k, threshold=threshold)
    elapsed = (time.perf_counter() - t0) * 1000
    metrics.round_num = 0
    metrics.elapsed_ms = elapsed
    rounds.append(metrics)
    print(
        f"  P@{top_k}: {metrics.precision:.3f}  "
        f"R@{top_k}: {metrics.recall:.3f}  "
        f"nDCG@{top_k}: {metrics.ndcg:.3f}  "
        f"[{elapsed:.0f}ms]"
    )

    if verbose:
        _print_per_query(details, top_k)

    # Learning rounds
    for rnd in range(1, learning_rounds + 1):
        print(f"\n--- Round {rnd}/{learning_rounds} ---")

        # Feed back results from previous round
        t0 = time.perf_counter()
        accepted, rejected = feed_back(ssm, details)
        print(f"  Feedback: {accepted} accepted, {rejected} rejected")

        if verbose:
            factors = ssm._interoception.factors.summary()
            print(
                f"  Factors: {factors['count']} nodes, "
                f"mean={factors['mean']:.3f}, "
                f"boosted={factors.get('boosted', 0)}, "
                f"penalized={factors.get('penalized', 0)}"
            )

        # Re-evaluate
        metrics, details = eval_round(ssm, top_k=top_k, threshold=threshold)
        elapsed = (time.perf_counter() - t0) * 1000
        metrics.round_num = rnd
        metrics.feedback_given = accepted + rejected
        metrics.accepted = accepted
        metrics.rejected = rejected
        metrics.elapsed_ms = elapsed

        delta_p = metrics.precision - rounds[-1].precision
        delta_r = metrics.recall - rounds[-1].recall
        rounds.append(metrics)

        print(
            f"  P@{top_k}: {metrics.precision:.3f} ({delta_p:+.3f})  "
            f"R@{top_k}: {metrics.recall:.3f} ({delta_r:+.3f})  "
            f"nDCG@{top_k}: {metrics.ndcg:.3f}  "
            f"[{elapsed:.0f}ms]"
        )

        if verbose:
            _print_per_query(details, top_k)

    # Summary
    _print_summary(rounds, top_k)
    return rounds


def _print_per_query(details: list[dict[str, Any]], top_k: int) -> None:
    """Print per-query breakdown."""
    for d in details:
        task = TASKS[d["task_idx"]][:60]
        print(
            f"    {task:<60} "
            f"P={d['precision']:.2f} R={d['recall']:.2f} "
            f"ret={len(d['retrieved'])} rel={len(d['relevant'])}"
        )


def _print_summary(rounds: list[RoundMetrics], top_k: int) -> None:
    """Print learning curve summary."""
    print("\n" + "=" * 65)
    print(f"Learning Curve (P@{top_k} / R@{top_k} / nDCG@{top_k})")
    print("=" * 65)
    print(f"{'Round':>6} {'P@k':>8} {'R@k':>8} {'nDCG@k':>8} {'dP':>8} {'dR':>8} {'Feedback':>10}")
    print("-" * 65)

    for r in rounds:
        dp = ""
        dr = ""
        if r.round_num > 0:
            prev = rounds[r.round_num - 1]
            dp = f"{r.precision - prev.precision:+.3f}"
            dr = f"{r.recall - prev.recall:+.3f}"
        print(
            f"{r.round_num:>6} "
            f"{r.precision:>8.3f} "
            f"{r.recall:>8.3f} "
            f"{r.ndcg:>8.3f} "
            f"{dp:>8} "
            f"{dr:>8} "
            f"{r.feedback_given:>10}"
        )

    print("-" * 65)
    if len(rounds) >= 2:
        # Peak metrics across all rounds
        best_p = max(rounds, key=lambda r: r.precision)
        best_r = max(rounds, key=lambda r: r.recall)
        best_n = max(rounds, key=lambda r: r.ndcg)
        p_peak = best_p.precision - rounds[0].precision
        r_peak = best_r.recall - rounds[0].recall
        n_peak = best_n.ndcg - rounds[0].ndcg
        print(
            f"Peak improvement:  P {p_peak:+.3f} (R{best_p.round_num})  "
            f"R {r_peak:+.3f} (R{best_r.round_num})  "
            f"nDCG {n_peak:+.3f} (R{best_n.round_num})"
        )
        # Final vs baseline
        p_final = rounds[-1].precision - rounds[0].precision
        r_final = rounds[-1].recall - rounds[0].recall
        n_final = rounds[-1].ndcg - rounds[0].ndcg
        print(f"Final vs baseline: P {p_final:+.3f}  R {r_final:+.3f}  nDCG {n_final:+.3f}")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoGen TCM retrieval eval with learning loop"
    )
    parser.add_argument("--rounds", type=int, default=3, help="Learning rounds")
    parser.add_argument("--top-k", type=int, default=5, help="Results per query")
    parser.add_argument("--threshold", type=float, default=1.7, help="Distance threshold")
    parser.add_argument("-v", "--verbose", action="store_true", help="Per-query details")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    run_eval(
        learning_rounds=args.rounds,
        top_k=args.top_k,
        threshold=args.threshold,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
