"""Edge pruning and classification for knowledge graph quality.

This module provides deterministic, LLM-free post-processing for extracted edges.
It is intentionally loosely coupled to qortex and can be used standalone.

=============================================================================
USE CASES
=============================================================================

1. ONLINE EXTRACTION (default)
   Run during ingestion to filter low-quality edges before graph storage.
   Enable/disable via flag. Keeps graph clean from the start.

2. CLI ANALYSIS
   `qortex prune --dry-run --min-confidence 0.55`
   Inspect what would be pruned without modifying data.
   Useful for threshold tuning and quality audits.

3. STATIC POST-PROCESSING
   Apply to existing graph data for cleanup or migration.
   Can re-run with different thresholds as quality standards evolve.

4. DYNAMIC AGENT USE (future)
   Agent can call pruning with variable "epsilon" based on internal state.
   Tighter pruning when high precision needed, looser for exploration.
   Enables intentional subgraph constraint based on context/affect.

=============================================================================
PRUNING PIPELINE (6 steps, all deterministic)
=============================================================================

Step 1: MINIMUM EVIDENCE LENGTH
  - Drop edges where source_text < 8 tokens
  - Rationale: Short quotes like "this helps" don't prove relationships.
    They're often extraction artifacts or vague implications.
    8 tokens approximates a minimal complete sentence fragment.

Step 2: CONFIDENCE FLOOR
  - Drop if confidence < 0.55 (hallucination zone)
  - Mark as "weak" if 0.55 <= confidence < 0.70
  - Mark as "strong" if confidence >= 0.70
  - Rationale: Below 0.55, the model itself is uncertain. These edges
    correlate with extraction errors. The weak/strong distinction enables
    downstream consumers to filter further if needed.

Step 3: JACCARD DEDUPLICATION
  - For edges with same (source_id, target_id, relation_type):
    Compute Jaccard similarity of source_text tokens.
  - If overlap > 0.6: keep only highest-confidence edge
  - Rationale: LLMs often emit the same relationship multiple times with
    slight rewording. We keep the best-evidenced version and drop redundancy.

Step 4: COMPETING RELATION RESOLUTION
  - For edges with same (source_id, target_id) but different relation_type:
    Check if source_text overlap < 0.3
  - If overlap >= 0.3: keep only higher confidence (same evidence, pick one)
  - If overlap < 0.3: keep both (genuinely different evidence)
  - Rationale: Prevents one quote from spawning contradictory edges
    (e.g., both SUPPORTS and CHALLENGES from same sentence).

Step 5: ISOLATED WEAK EDGE PRUNING
  - If a concept has only 1 edge AND that edge is weak (conf < 0.65):
    Drop the edge
  - Rationale: Singleton weak connections are often noise. Well-connected
    concepts can tolerate some weak edges; isolated ones cannot.

Step 6: STRUCTURAL VS CAUSAL TAGGING
  - Tag each surviving edge with layer = "structural" or "causal"
  - Not pruning, but classification for downstream reasoning.

=============================================================================
STRUCTURAL VS CAUSAL CLASSIFICATION
=============================================================================

STRUCTURAL RELATIONS (Map Layer):
  - PART_OF: Compositional structure
  - REFINES: Taxonomic specialization
  - IMPLEMENTS: Concrete realization of abstraction
  - SIMILAR_TO: Lateral conceptual proximity
  - ALTERNATIVE_TO: Design-space branching
  - USES: Functional dependency (non-causal, static)

These describe WHAT THINGS ARE. They define the shape of the knowledge space.
They answer: "How is this domain organized?"

CAUSAL RELATIONS (Causal Layer):
  - REQUIRES: Existence/operation dependency
  - SUPPORTS: Provides rationale or enabling force
  - CHALLENGES: Introduces tension or counterforce
  - CONTRADICTS: Logical or conceptual incompatibility

These describe INFLUENCE AND JUSTIFICATION. They support reasoning.
They answer: "What leads to what, and why?"

This separation enables:
  - Structural layer: graph layout, clustering, ontology, similarity search
  - Causal layer: reasoning paths, explanation, tradeoff analysis
  - Binding (future): structural neighbors constrain causal plausibility

=============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# Relation type classifications
STRUCTURAL_RELATIONS = frozenset({
    "part_of", "refines", "implements",
    "similar_to", "alternative_to", "uses",
})

CAUSAL_RELATIONS = frozenset({
    "requires", "supports", "challenges", "contradicts",
})


@dataclass
class PruningConfig:
    """Configuration for edge pruning pipeline.

    All thresholds have been chosen based on empirical observation
    of LLM extraction behavior. See module docstring for rationale.
    """
    # Step 1: Minimum evidence
    min_evidence_tokens: int = 8

    # Step 2: Confidence thresholds
    confidence_floor: float = 0.55  # Below this = drop
    confidence_weak: float = 0.70   # Below this = weak, above = strong

    # Step 3: Jaccard dedup
    jaccard_duplicate_threshold: float = 0.6

    # Step 4: Competing relation resolution
    competing_overlap_threshold: float = 0.3

    # Step 5: Isolated weak edge
    isolated_weak_confidence: float = 0.65

    # Control flags
    enabled: bool = True
    tag_layers: bool = True  # Add structural/causal classification


@dataclass
class PruningResult:
    """Result of pruning pipeline."""
    edges: list[dict] = field(default_factory=list)  # Surviving edges

    # Statistics
    input_count: int = 0
    dropped_low_evidence: int = 0
    dropped_low_confidence: int = 0
    dropped_duplicate: int = 0
    dropped_competing: int = 0
    dropped_isolated: int = 0

    # Surviving edge classification
    strong_count: int = 0
    weak_count: int = 0
    structural_count: int = 0
    causal_count: int = 0

    @property
    def output_count(self) -> int:
        return len(self.edges)

    @property
    def total_dropped(self) -> int:
        return self.input_count - self.output_count

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Pruning: {self.input_count} -> {self.output_count} edges",
            f"  Dropped: {self.total_dropped}",
            f"    - Low evidence: {self.dropped_low_evidence}",
            f"    - Low confidence: {self.dropped_low_confidence}",
            f"    - Duplicates: {self.dropped_duplicate}",
            f"    - Competing: {self.dropped_competing}",
            f"    - Isolated weak: {self.dropped_isolated}",
            f"  Surviving: {self.strong_count} strong, {self.weak_count} weak",
            f"  Layers: {self.structural_count} structural, {self.causal_count} causal",
        ]
        return "\n".join(lines)


def tokenize(text: str) -> set[str]:
    """Simple tokenization for Jaccard similarity.

    Lowercase, strip punctuation, remove common stopwords.
    """
    if not text:
        return set()

    # Lowercase and extract words
    words = re.findall(r'\b[a-z]+\b', text.lower())

    # Remove common stopwords
    stopwords = {
        'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
        'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
        'would', 'could', 'should', 'may', 'might', 'must', 'shall',
        'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'under', 'again', 'further',
        'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
        'all', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
        'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
        'very', 'just', 'and', 'but', 'if', 'or', 'because', 'until',
        'while', 'this', 'that', 'these', 'those', 'it', 'its',
    }

    return {w for w in words if w not in stopwords}


def jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute Jaccard similarity between token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def classify_layer(relation_type: str) -> Literal["structural", "causal"]:
    """Classify relation type as structural or causal."""
    rel_lower = relation_type.lower()
    if rel_lower in STRUCTURAL_RELATIONS:
        return "structural"
    elif rel_lower in CAUSAL_RELATIONS:
        return "causal"
    else:
        # Default unknown relations to structural (safer)
        return "structural"


def prune_edges(
    edges: list[dict],
    config: PruningConfig | None = None,
) -> PruningResult:
    """Apply full pruning pipeline to edge list.

    Args:
        edges: List of edge dicts with at minimum:
            - source_id, target_id, relation_type, confidence
            - source_text (optional but recommended)
        config: Pruning configuration. Uses defaults if None.

    Returns:
        PruningResult with surviving edges and statistics.
    """
    if config is None:
        config = PruningConfig()

    if not config.enabled:
        # Pruning disabled, just tag layers
        result_edges = []
        for edge in edges:
            edge = dict(edge)  # Copy
            if config.tag_layers:
                edge["layer"] = classify_layer(edge.get("relation_type", ""))
            result_edges.append(edge)
        return PruningResult(
            edges=result_edges,
            input_count=len(edges),
            structural_count=sum(1 for e in result_edges if e.get("layer") == "structural"),
            causal_count=sum(1 for e in result_edges if e.get("layer") == "causal"),
        )

    result = PruningResult(input_count=len(edges))

    # Precompute tokens for each edge
    edge_tokens: dict[int, set[str]] = {}
    for i, edge in enumerate(edges):
        edge_tokens[i] = tokenize(edge.get("source_text", ""))

    # Step 1: Minimum evidence length
    surviving = []
    for i, edge in enumerate(edges):
        tokens = edge_tokens[i]
        if len(tokens) < config.min_evidence_tokens:
            result.dropped_low_evidence += 1
        else:
            surviving.append((i, edge))

    # Step 2: Confidence floor
    after_confidence = []
    for i, edge in surviving:
        conf = edge.get("confidence", 0.0)
        if conf < config.confidence_floor:
            result.dropped_low_confidence += 1
        else:
            # Mark strength
            edge = dict(edge)
            edge["strength"] = "strong" if conf >= config.confidence_weak else "weak"
            after_confidence.append((i, edge))

    # Step 3: Jaccard deduplication
    # Group by (source_id, target_id, relation_type)
    from collections import defaultdict
    groups: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    for i, edge in after_confidence:
        key = (edge["source_id"], edge["target_id"], edge["relation_type"])
        groups[key].append((i, edge))

    after_dedup = []
    for key, group in groups.items():
        if len(group) == 1:
            after_dedup.append(group[0])
        else:
            # Check Jaccard similarity, keep highest confidence
            kept = []
            for i, edge in sorted(group, key=lambda x: -x[1].get("confidence", 0)):
                tokens_i = edge_tokens[i]
                is_dup = False
                for j, kept_edge in kept:
                    tokens_j = edge_tokens[j]
                    if jaccard_similarity(tokens_i, tokens_j) > config.jaccard_duplicate_threshold:
                        is_dup = True
                        result.dropped_duplicate += 1
                        break
                if not is_dup:
                    kept.append((i, edge))
            after_dedup.extend(kept)

    # Step 4: Competing relation resolution
    # Group by (source_id, target_id) only
    pair_groups: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    for i, edge in after_dedup:
        key = (edge["source_id"], edge["target_id"])
        pair_groups[key].append((i, edge))

    after_competing = []
    for key, group in pair_groups.items():
        if len(group) == 1:
            after_competing.append(group[0])
        else:
            # Check for competing relations with overlapping evidence
            kept = []
            for i, edge in sorted(group, key=lambda x: -x[1].get("confidence", 0)):
                tokens_i = edge_tokens[i]
                should_drop = False
                for j, kept_edge in kept:
                    if edge["relation_type"] != kept_edge["relation_type"]:
                        tokens_j = edge_tokens[j]
                        if jaccard_similarity(tokens_i, tokens_j) >= config.competing_overlap_threshold:
                            should_drop = True
                            result.dropped_competing += 1
                            break
                if not should_drop:
                    kept.append((i, edge))
            after_competing.extend(kept)

    # Step 5: Isolated weak edge pruning
    # Count degrees
    from collections import Counter
    degree: Counter[str] = Counter()
    for i, edge in after_competing:
        degree[edge["source_id"]] += 1
        degree[edge["target_id"]] += 1

    after_isolated = []
    for i, edge in after_competing:
        src_degree = degree[edge["source_id"]]
        tgt_degree = degree[edge["target_id"]]

        # Check if either endpoint is isolated (degree 1)
        # and the edge is weak
        is_isolated = (src_degree == 1 or tgt_degree == 1)
        is_weak = edge.get("confidence", 0) < config.isolated_weak_confidence

        if is_isolated and is_weak:
            result.dropped_isolated += 1
        else:
            after_isolated.append((i, edge))

    # Step 6: Layer tagging
    final_edges = []
    for i, edge in after_isolated:
        edge = dict(edge)
        if config.tag_layers:
            edge["layer"] = classify_layer(edge.get("relation_type", ""))
        final_edges.append(edge)

    # Compute statistics
    result.edges = final_edges
    result.strong_count = sum(1 for e in final_edges if e.get("strength") == "strong")
    result.weak_count = sum(1 for e in final_edges if e.get("strength") == "weak")
    result.structural_count = sum(1 for e in final_edges if e.get("layer") == "structural")
    result.causal_count = sum(1 for e in final_edges if e.get("layer") == "causal")

    return result


def prune_edges_dry_run(
    edges: list[dict],
    config: PruningConfig | None = None,
) -> PruningResult:
    """Analyze what pruning would do without modifying edges.

    Same as prune_edges but returns original edges (unmodified)
    along with statistics of what WOULD be dropped.
    """
    result = prune_edges(edges, config)
    # Return original edges but keep the statistics
    result.edges = list(edges)
    return result
