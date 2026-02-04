"""Tests for core data models."""

import pytest
from qortex.core.models import (
    ConceptEdge,
    ConceptNode,
    ExplicitRule,
    IngestionManifest,
    RelationType,
    SourceMetadata,
)


def test_concept_node_creation():
    """ConceptNode can be created with required fields."""
    node = ConceptNode(
        id="fp_js:pure_function",
        name="Pure Function",
        description="A function with no side effects",
        domain="fp_js",
        source_id="fp_js:book",
    )

    assert node.id == "fp_js:pure_function"
    assert node.domain == "fp_js"
    assert node.confidence == 1.0  # default


def test_concept_edge_creation():
    """ConceptEdge can be created with relationship type."""
    edge = ConceptEdge(
        source_id="fp_js:pure_function",
        target_id="fp_js:side_effect",
        relation_type=RelationType.CONTRADICTS,
    )

    assert edge.relation_type == RelationType.CONTRADICTS
    assert edge.confidence == 1.0


def test_ingestion_manifest():
    """IngestionManifest bundles extraction results."""
    source = SourceMetadata(
        id="test_source",
        name="Test Book",
        source_type="pdf",
        path_or_url="/path/to/book.pdf",
    )

    concept = ConceptNode(
        id="test:concept",
        name="Test Concept",
        description="A test concept",
        domain="test",
        source_id="test_source",
    )

    rule = ExplicitRule(
        id="test:rule:0",
        text="Always test your code",
        domain="test",
        source_id="test_source",
    )

    manifest = IngestionManifest(
        source=source,
        domain="test",
        concepts=[concept],
        edges=[],
        rules=[rule],
    )

    assert manifest.domain == "test"
    assert len(manifest.concepts) == 1
    assert len(manifest.rules) == 1


def test_relation_types():
    """All expected relation types exist."""
    assert RelationType.CONTRADICTS.value == "contradicts"
    assert RelationType.REQUIRES.value == "requires"
    assert RelationType.REFINES.value == "refines"
    assert RelationType.IMPLEMENTS.value == "implements"
