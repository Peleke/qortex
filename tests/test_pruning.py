"""Tests for edge pruning module."""

import pytest

from qortex.core.pruning import (
    CAUSAL_RELATIONS,
    STRUCTURAL_RELATIONS,
    PruningConfig,
    classify_layer,
    jaccard_similarity,
    prune_edges,
    prune_edges_dry_run,
    tokenize,
)


class TestTokenize:
    def test_basic_tokenization(self):
        tokens = tokenize("The quick brown fox jumps over the lazy dog.")
        assert "quick" in tokens
        assert "brown" in tokens
        assert "fox" in tokens
        assert "jumps" in tokens
        # Stopwords removed
        assert "the" not in tokens
        assert "and" not in tokenize("foo and bar")

    def test_empty_string(self):
        assert tokenize("") == set()

    def test_none_handling(self):
        assert tokenize(None) == set()

    def test_punctuation_stripped(self):
        tokens = tokenize("Hello, world! This is a test.")
        assert "hello" in tokens
        assert "world" in tokens
        assert "," not in tokens


class TestJaccardSimilarity:
    def test_identical_sets(self):
        s = {"a", "b", "c"}
        assert jaccard_similarity(s, s) == 1.0

    def test_disjoint_sets(self):
        assert jaccard_similarity({"a", "b"}, {"c", "d"}) == 0.0

    def test_partial_overlap(self):
        # {a, b, c} and {b, c, d} -> intersection {b, c}, union {a, b, c, d}
        # Jaccard = 2/4 = 0.5
        assert jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"}) == 0.5

    def test_empty_sets(self):
        assert jaccard_similarity(set(), set()) == 0.0
        assert jaccard_similarity({"a"}, set()) == 0.0


class TestClassifyLayer:
    def test_structural_relations(self):
        for rel in STRUCTURAL_RELATIONS:
            assert classify_layer(rel) == "structural"
            assert classify_layer(rel.upper()) == "structural"

    def test_causal_relations(self):
        for rel in CAUSAL_RELATIONS:
            assert classify_layer(rel) == "causal"
            assert classify_layer(rel.upper()) == "causal"

    def test_unknown_defaults_to_structural(self):
        assert classify_layer("unknown_relation") == "structural"


class TestPruneEdges:
    @pytest.fixture
    def sample_edges(self):
        return [
            {
                "source_id": "a",
                "target_id": "b",
                "relation_type": "REQUIRES",
                "confidence": 0.9,
                "source_text": "Concept A fundamentally requires concept B to function properly and correctly in the production system.",
            },
            {
                "source_id": "b",
                "target_id": "c",
                "relation_type": "SUPPORTS",
                "confidence": 0.85,
                "source_text": "Concept B provides strong evidence and comprehensive support for the validity and correctness of C.",
            },
            {
                "source_id": "c",
                "target_id": "d",
                "relation_type": "PART_OF",
                "confidence": 0.4,  # Low confidence - should be dropped
                "source_text": "Concept C is an integral component and constituent part of the larger concept D in this domain.",
            },
            {
                "source_id": "d",
                "target_id": "e",
                "relation_type": "USES",
                "confidence": 0.75,
                "source_text": "short",  # Too short - should be dropped
            },
        ]

    def test_drops_low_confidence(self, sample_edges):
        result = prune_edges(sample_edges)
        confidences = [e.get("confidence") for e in result.edges]
        assert all(c >= 0.55 for c in confidences)
        assert result.dropped_low_confidence >= 1

    def test_drops_low_evidence(self, sample_edges):
        result = prune_edges(sample_edges)
        assert result.dropped_low_evidence >= 1

    def test_tags_layers(self, sample_edges):
        result = prune_edges(sample_edges)
        for edge in result.edges:
            assert "layer" in edge
            assert edge["layer"] in ("structural", "causal")

    def test_tags_strength(self, sample_edges):
        result = prune_edges(sample_edges)
        for edge in result.edges:
            assert "strength" in edge
            assert edge["strength"] in ("strong", "weak")

    def test_disabled_pruning_still_tags(self, sample_edges):
        config = PruningConfig(enabled=False)
        result = prune_edges(sample_edges, config)
        # All edges should survive
        assert result.output_count == len(sample_edges)
        # But should still be tagged
        for edge in result.edges:
            assert "layer" in edge

    def test_summary_output(self, sample_edges):
        result = prune_edges(sample_edges)
        summary = result.summary()
        assert "Pruning:" in summary
        assert "Dropped:" in summary


class TestJaccardDedup:
    def test_deduplicates_similar_evidence(self):
        edges = [
            {
                "source_id": "a",
                "target_id": "b",
                "relation_type": "REQUIRES",
                "confidence": 0.9,
                "source_text": "Concept A fundamentally requires concept B to function correctly and properly in the production system environment.",
            },
            {
                "source_id": "a",
                "target_id": "b",
                "relation_type": "REQUIRES",
                "confidence": 0.8,  # Lower confidence, similar text
                "source_text": "Concept A fundamentally requires concept B to function properly and correctly in the production system environment.",
            },
        ]
        result = prune_edges(edges)
        # Should keep only one (the higher confidence one)
        assert result.output_count == 1
        assert result.edges[0]["confidence"] == 0.9
        assert result.dropped_duplicate == 1


class TestCompetingRelations:
    def test_keeps_distinct_evidence(self):
        edges = [
            {
                "source_id": "a",
                "target_id": "b",
                "relation_type": "SUPPORTS",
                "confidence": 0.85,
                "source_text": "Concept A provides substantial evidence for concept B through the novel mechanism X described in the literature.",
            },
            {
                "source_id": "a",
                "target_id": "b",
                "relation_type": "CHALLENGES",
                "confidence": 0.80,
                "source_text": "However concept A also introduces significant problems for concept B via the alternative pathway Y and Z.",
            },
        ]
        result = prune_edges(edges)
        # Both should survive (different evidence)
        assert result.output_count == 2

    def test_resolves_overlapping_evidence(self):
        edges = [
            {
                "source_id": "a",
                "target_id": "b",
                "relation_type": "SUPPORTS",
                "confidence": 0.85,
                "source_text": "Concept A supports concept B through this particular mechanism in the software design context.",
            },
            {
                "source_id": "a",
                "target_id": "b",
                "relation_type": "CHALLENGES",
                "confidence": 0.75,
                "source_text": "Concept A supports concept B through this same particular mechanism in software design context.",
            },
        ]
        result = prune_edges(edges)
        # Should keep only the higher confidence one
        assert result.output_count == 1
        assert result.edges[0]["relation_type"] == "SUPPORTS"


class TestDryRun:
    def test_dry_run_returns_original_edges(self):
        edges = [
            {
                "source_id": "a",
                "target_id": "b",
                "relation_type": "REQUIRES",
                "confidence": 0.3,  # Would be dropped
                "source_text": "A requires B.",
            },
        ]
        result = prune_edges_dry_run(edges)
        # Original edge should be in output
        assert len(result.edges) == 1
        # But stats should show it would be dropped
        assert result.dropped_low_confidence > 0 or result.dropped_low_evidence > 0
