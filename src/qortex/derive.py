"""Rule derivation from enriched knowledge graph patterns.

Two derivation patterns:
1. Uncovered mistake clusters — mistakes with no path to credited rules
2. Cross-domain bridges — embedding similarity finds rules from other domains

Uses CausalDAG for structural reasoning and sqlite-vec for similarity.
LLM cascade (Ollama → API → template) for rule text generation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# Bridge threshold protocol (designed for future RL-learned values)
# =============================================================================


class BridgeThreshold(Protocol):
    """Returns similarity threshold for cross-domain bridging.

    Takes (source_domain, target_domain) because the optimal threshold
    differs per domain pair. Static for now; future implementations
    wrap a contextual bandit.
    """

    def get_threshold(self, source_domain: str, target_domain: str) -> float: ...


@dataclass
class StaticThreshold:
    """Fixed threshold — the noop implementation."""

    value: float = 0.75

    def get_threshold(self, source_domain: str, target_domain: str) -> float:
        return self.value


# =============================================================================
# Derived rule output
# =============================================================================


@dataclass
class DerivedRule:
    """A rule derived from graph pattern analysis."""

    text: str
    category: str
    derivation_type: str  # "uncovered_cluster" or "cross_domain_bridge"
    confidence: float
    provenance: dict[str, Any] = field(default_factory=dict)
    source_domain: str = "buildlog"
    target_domain: str = "buildlog"


# =============================================================================
# LLM cascade for rule generation
# =============================================================================


def _generate_rule_llm(
    descriptions: list[str],
    error_class: str,
    resolution_actions: list[str],
) -> str | None:
    """Try to generate a rule via LLM cascade: Ollama → Anthropic → None.

    Returns rule text or None if no LLM available.
    """
    prompt = (
        "You are analyzing a cluster of recurring development mistakes.\n"
        f"Error class: {error_class}\n\n"
        "Mistake descriptions:\n"
        + "\n".join(f"- {d}" for d in descriptions[:10])
        + "\n\nResolution actions taken:\n"
        + "\n".join(f"- {r}" for r in resolution_actions[:10] if r)
        + "\n\nDerive ONE concise, actionable rule (1-2 sentences) that would "
        "prevent this class of mistakes. The rule should be specific enough "
        "to be useful but general enough to apply across similar situations. "
        "Return only the rule text, nothing else."
    )

    # Try Ollama first
    try:
        import ollama

        model = os.environ.get("QORTEX_OLLAMA_MODEL", "qwen2.5:14b")
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp["message"]["content"].strip()
        if text:
            logger.info("Rule generated via Ollama")
            return text
    except Exception as e:
        logger.debug("Ollama unavailable: %s", e)

    # Try Anthropic
    try:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        if text:
            logger.info("Rule generated via Anthropic")
            return text
    except Exception as e:
        logger.debug("Anthropic unavailable: %s", e)

    return None


def _generate_rule_template(
    error_class: str,
    resolution_actions: list[str],
) -> str:
    """Template-based rule generation — the last-resort fallback."""
    if resolution_actions:
        # Use most common resolution
        from collections import Counter

        most_common = Counter(r for r in resolution_actions if r).most_common(1)
        if most_common:
            return f"When encountering {error_class} errors, {most_common[0][0]}"
    return f"Verify assumptions before {error_class} operations to prevent recurring mistakes"


def generate_rule_text(
    descriptions: list[str],
    error_class: str,
    resolution_actions: list[str],
) -> tuple[str, str]:
    """Generate rule text via LLM cascade with template fallback.

    Returns:
        Tuple of (rule_text, generation_method).
    """
    llm_text = _generate_rule_llm(descriptions, error_class, resolution_actions)
    if llm_text:
        return llm_text, "llm"

    return _generate_rule_template(error_class, resolution_actions), "template"


# =============================================================================
# Pattern 1: Uncovered mistake clusters
# =============================================================================
#
# Algorithm (leverages qortex CausalDAG):
#
# 1. Build CausalDAG from the KG after ingestion
# 2. Identify credited rule nodes (from gauntlet_credits)
# 3. For each mistake node, trace FORWARD through the DAG —
#    if no causal path reaches any credited rule, the mistake
#    is truly uncovered (no rule in the system addresses it)
# 4. Group uncovered mistakes by error_class → clusters
# 5. Per cluster: LLM cascade generates a covering rule
# 6. CreditAssigner.to_posterior_updates() provides the exact
#    Thompson Sampling deltas for the article visualization
#
# Why CausalDAG instead of naive edge checking:
#   A mistake might not have a direct CHALLENGES edge to a rule,
#   but a causal ancestor (e.g., a concept it PART_OF) might.
#   The DAG catches transitive coverage that edge-checking misses.
#   This also gives us the causal PATH for provenance — showing
#   exactly why a rule does or doesn't apply.
# =============================================================================


def find_uncovered_mistakes(
    backend: Any,
    credit_stats: dict[str, Any] | None = None,
    domain: str = "buildlog",
    use_causal_dag: bool = True,
) -> dict[str, list[dict]]:
    """Find mistake nodes not covered by any credited rule.

    When use_causal_dag=True (default), builds a CausalDAG and traces
    forward from each mistake — a mistake is covered only if some
    causal path reaches a credited rule node. This catches transitive
    coverage that direct edge checking misses.

    Args:
        backend: GraphBackend (Memgraph or in-memory).
        credit_stats: RuleCreditStats dict (if None, ALL rules are uncredited).
        domain: Domain to search.
        use_causal_dag: Use CausalDAG path tracing (True) or naive edges (False).

    Returns:
        Dict of error_class -> list of mistake node property dicts.
    """
    credited_rule_ids = set()
    if credit_stats:
        credited_rule_ids = {
            f"gauntlet_rule:{rid}" for rid in credit_stats
        }

    # Build CausalDAG for structural path tracing
    dag = None
    if use_causal_dag:
        try:
            from qortex.causal.dag import CausalDAG

            dag = CausalDAG.from_backend(backend, domain)
            logger.info(
                "CausalDAG built: %d nodes, %d edges",
                len(dag.node_ids), dag.edge_count,
            )
        except Exception as e:
            logger.debug("CausalDAG unavailable, falling back to edge checking: %s", e)

    uncovered: dict[str, list[dict]] = {}

    for node in backend.find_nodes(domain=domain, limit=100_000):
        if not node.source_id or not node.source_id.startswith("buildlog:mistake"):
            continue

        covered = False

        if dag is not None and node.id in dag.node_ids:
            # CausalDAG path tracing: check if any descendant is a credited rule
            descendants = dag.descendants(node.id)
            if descendants & credited_rule_ids:
                covered = True
        else:
            # Fallback: direct edge checking
            for edge in backend.get_edges(node.id, direction="out"):
                rel = (
                    edge.relation_type.value
                    if hasattr(edge.relation_type, "value")
                    else str(edge.relation_type)
                )
                if rel == "challenges" and edge.target_id in credited_rule_ids:
                    covered = True
                    break

        if not covered:
            error_class = node.properties.get("error_class", "unknown")
            if error_class not in uncovered:
                uncovered[error_class] = []

            # Include causal context in provenance
            causal_context = {}
            if dag is not None and node.id in dag.node_ids:
                desc = dag.descendants(node.id)
                causal_context = {
                    "dag_descendants": len(desc),
                    "nearest_rules": sorted(
                        [d for d in desc if d.startswith("gauntlet_rule:")]
                    )[:5],
                }

            uncovered[error_class].append({
                "id": node.id,
                "description": node.description,
                "error_class": error_class,
                "resolution_action": node.properties.get("resolution_action", ""),
                "causal_context": causal_context,
            })

    return uncovered


def cluster_mistakes(
    mistakes: list[dict],
    vector_index: Any | None = None,
    embedding_model: Any | None = None,
    min_cluster_size: int = 2,
) -> list[list[dict]]:
    """Cluster mistakes by semantic similarity.

    If vector_index/embedding_model available, uses embedding clustering.
    Otherwise falls back to grouping by error_class (already done by caller).

    Args:
        mistakes: List of mistake dicts (from find_uncovered_mistakes).
        vector_index: SqliteVecIndex or NumpyVectorIndex (optional).
        embedding_model: SentenceTransformerEmbedding (optional).
        min_cluster_size: Minimum mistakes per cluster.

    Returns:
        List of clusters, each a list of mistake dicts.
    """
    if len(mistakes) < min_cluster_size:
        return []

    # If we have embeddings, cluster by similarity
    if embedding_model is not None:
        try:
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering

            texts = [m["description"] for m in mistakes]
            embeddings = np.array(embedding_model.embed(texts), dtype=np.float32)

            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                metric="cosine",
                linkage="average",
            )
            labels = clustering.fit_predict(embeddings)

            clusters: dict[int, list[dict]] = {}
            for label, mistake in zip(labels, mistakes):
                clusters.setdefault(label, []).append(mistake)

            return [c for c in clusters.values() if len(c) >= min_cluster_size]

        except ImportError:
            logger.debug("sklearn not available, falling back to single cluster")

    # Fallback: treat all mistakes in this error_class as one cluster
    if len(mistakes) >= min_cluster_size:
        return [mistakes]
    return []


def derive_from_uncovered_clusters(
    backend: Any,
    credit_stats: dict[str, Any] | None = None,
    domain: str = "buildlog",
    min_cluster_size: int = 2,
    embedding_model: Any | None = None,
) -> list[DerivedRule]:
    """Pattern 1: Derive rules from uncovered mistake clusters.

    1. Find mistakes with no CHALLENGES edge to credited rules
    2. Group by error_class
    3. Sub-cluster by embedding similarity (if available)
    4. Generate one rule per cluster via LLM cascade

    Returns:
        List of DerivedRule with provenance.
    """
    uncovered = find_uncovered_mistakes(backend, credit_stats, domain)
    if not uncovered:
        logger.info("No uncovered mistake clusters found")
        return []

    derived: list[DerivedRule] = []

    for error_class, mistakes in uncovered.items():
        clusters = cluster_mistakes(
            mistakes,
            embedding_model=embedding_model,
            min_cluster_size=min_cluster_size,
        )

        for cluster in clusters:
            descriptions = [m["description"] for m in cluster]
            resolutions = [m["resolution_action"] for m in cluster if m.get("resolution_action")]

            rule_text, method = generate_rule_text(descriptions, error_class, resolutions)

            derived.append(DerivedRule(
                text=rule_text,
                category=error_class,
                derivation_type="uncovered_cluster",
                confidence=min(0.5 + len(cluster) * 0.05, 0.9),
                provenance={
                    "type": "uncovered_cluster",
                    "error_class": error_class,
                    "mistake_count": len(cluster),
                    "mistake_ids": [m["id"] for m in cluster],
                    "generation_method": method,
                },
            ))

    logger.info("Derived %d rules from %d uncovered clusters", len(derived), len(uncovered))
    return derived


# =============================================================================
# Pattern 2: Cross-domain bridges
# =============================================================================


def derive_cross_domain_bridges(
    backend: Any,
    uncovered_clusters: dict[str, list[dict]],
    vector_index: Any,
    embedding_model: Any,
    threshold: BridgeThreshold | None = None,
    source_domain: str = "buildlog",
) -> list[DerivedRule]:
    """Pattern 2: Find rules from other domains that cover uncovered mistakes.

    1. Compute centroid embedding for each uncovered cluster
    2. Vector search against ALL rule embeddings
    3. Filter: only rules from different domains
    4. If similarity > threshold: create bridge rule

    Args:
        backend: GraphBackend for node lookups.
        uncovered_clusters: From find_uncovered_mistakes().
        vector_index: Index with embedded rule descriptions.
        embedding_model: For embedding cluster centroids.
        threshold: BridgeThreshold protocol (default: StaticThreshold(0.75)).
        source_domain: Domain of the uncovered mistakes.

    Returns:
        List of DerivedRule with cross-domain provenance.
    """
    import asyncio

    import numpy as np

    if threshold is None:
        threshold = StaticThreshold()

    derived: list[DerivedRule] = []

    for error_class, mistakes in uncovered_clusters.items():
        if not mistakes:
            continue

        # Compute centroid of cluster descriptions
        descriptions = [m["description"] for m in mistakes]
        embeddings = np.array(embedding_model.embed(descriptions), dtype=np.float32)
        centroid = embeddings.mean(axis=0).tolist()

        # Vector search for similar nodes (cast wide, filter by domain)
        try:
            results = asyncio.run(vector_index.search(centroid, k=20))
        except Exception as e:
            logger.debug("Vector search failed for %s: %s", error_class, e)
            continue

        for node_id, similarity in results:
            # Skip nodes in the same domain
            try:
                node = backend.get_node(node_id)
            except Exception:
                continue

            if node is None or node.domain == source_domain:
                continue

            # Check threshold for this domain pair
            thresh = threshold.get_threshold(source_domain, node.domain)
            if similarity < thresh:
                continue

            # This is a bridge: a rule from another domain applies
            derived.append(DerivedRule(
                text=f"Apply: {node.description}",
                category=error_class,
                derivation_type="cross_domain_bridge",
                confidence=round(similarity, 4),
                source_domain=node.domain,
                target_domain=source_domain,
                provenance={
                    "type": "cross_domain_bridge",
                    "source_rule_id": node.id,
                    "source_domain": node.domain,
                    "source_description": node.description,
                    "similarity": round(similarity, 4),
                    "error_class": error_class,
                    "mistake_count": len(mistakes),
                    "mistake_ids": [m["id"] for m in mistakes],
                },
            ))
            logger.info(
                "Cross-domain bridge: %s → %s (sim=%.3f)",
                node.domain, error_class, similarity,
            )

    logger.info("Derived %d cross-domain bridge rules", len(derived))
    return derived


# =============================================================================
# Combined derivation
# =============================================================================


def derive_rules_from_graph(
    backend: Any,
    credit_stats: dict[str, Any] | None = None,
    vector_index: Any | None = None,
    embedding_model: Any | None = None,
    threshold: BridgeThreshold | None = None,
    domain: str = "buildlog",
    min_cluster_size: int = 2,
) -> list[DerivedRule]:
    """Derive rules from enriched KG using both patterns.

    Pattern 1 (always): Uncovered mistake clusters → LLM-generated rules
    Pattern 2 (if embeddings available): Cross-domain bridges via similarity

    Args:
        backend: GraphBackend with ingested emissions + credits.
        credit_stats: From read_gauntlet_credits() (for determining coverage).
        vector_index: SqliteVecIndex with embedded concepts (for pattern 2).
        embedding_model: Embedding model (for clustering + pattern 2).
        threshold: BridgeThreshold for cross-domain similarity.
        domain: Source domain.
        min_cluster_size: Minimum mistakes to form a cluster.

    Returns:
        Combined list of DerivedRule from both patterns.
    """
    rules: list[DerivedRule] = []

    # Pattern 1: Uncovered clusters
    p1_rules = derive_from_uncovered_clusters(
        backend,
        credit_stats=credit_stats,
        domain=domain,
        min_cluster_size=min_cluster_size,
        embedding_model=embedding_model,
    )
    rules.extend(p1_rules)

    # Pattern 2: Cross-domain bridges (only if embeddings available)
    if vector_index is not None and embedding_model is not None:
        uncovered = find_uncovered_mistakes(backend, credit_stats, domain)
        if uncovered:
            p2_rules = derive_cross_domain_bridges(
                backend,
                uncovered,
                vector_index,
                embedding_model,
                threshold=threshold,
                source_domain=domain,
            )
            rules.extend(p2_rules)

    logger.info(
        "Total derived: %d rules (%d cluster, %d bridge)",
        len(rules),
        len(p1_rules),
        len(rules) - len(p1_rules),
    )
    return rules


# =============================================================================
# Export to buildlog seed format
# =============================================================================


def export_derived_rules(
    rules: list[DerivedRule],
    persona: str = "qortex_derived",
) -> dict[str, Any]:
    """Serialize derived rules to buildlog seed format.

    The output is compatible with qortex's write_seed_to_pending(),
    which writes YAML that buildlog's ingest_seeds can consume.

    Args:
        rules: Derived rules to export.
        persona: Persona name for the seed file.

    Returns:
        Seed dict ready for write_seed_to_pending().
    """
    from datetime import UTC, datetime

    seed_rules = []
    for r in rules:
        seed_rules.append({
            "rule": r.text,
            "category": r.category,
            "context": f"Derived from {r.derivation_type}",
            "antipattern": "",
            "rationale": f"KG derivation ({r.derivation_type}): {r.provenance.get('generation_method', r.provenance.get('type', 'unknown'))}",
            "tags": ["kg-derived", r.derivation_type],
            "provenance": {
                "domain": r.source_domain,
                "derivation": "derived",
                "confidence": r.confidence,
                **{k: v for k, v in r.provenance.items()
                   if k not in ("type",)},
            },
        })

    return {
        "persona": persona,
        "version": 1,
        "rules": seed_rules,
        "metadata": {
            "source": "qortex",
            "source_version": "0.9.0",
            "projected_at": datetime.now(UTC).isoformat(),
            "rule_count": len(seed_rules),
            "derivation_patterns": list({r.derivation_type for r in rules}),
        },
    }
