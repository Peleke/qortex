"""Tests for illustrative vs generalizable concept classification.

Verifies the two-pass concept processing in Ingestor.ingest():
1. Generalizable concepts become standalone ConceptNodes
2. Illustrative concepts are routed to parent.properties["examples"]
3. Orphaned illustrative concepts fall back to tagged ConceptNodes
"""

from __future__ import annotations

from qortex_ingest.base import Chunk, Ingestor, Source, StubLLMBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleIngestor(Ingestor):
    """Ingestor with fixed chunks for testing concept processing."""

    def __init__(self, llm, chunks: list[Chunk] | None = None):
        super().__init__(llm)
        self._chunks = chunks or [Chunk(id="ch1", content="test content", location="ch1")]

    def chunk(self, source: Source) -> list[Chunk]:
        return self._chunks


def _make_source(name: str = "test") -> Source:
    return Source(raw_content="test content", source_type="text", name=name)


# ===========================================================================
# TestConceptRoleClassification
# ===========================================================================


class TestConceptRoleClassification:
    """Core classification: generalizable vs illustrative routing."""

    def test_generalizable_concept_becomes_node(self):
        """Standard generalizable concept → ConceptNode as before."""
        llm = StubLLMBackend(
            concepts=[
                {
                    "name": "Observer Pattern",
                    "description": "Pub-sub for objects",
                    "confidence": 0.9,
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        assert len(manifest.concepts) == 1
        assert manifest.concepts[0].name == "Observer Pattern"
        assert manifest.concepts[0].id == "design:Observer Pattern"

    def test_illustrative_concept_attached_to_parent(self):
        """Illustrative concept with known parent → parent.properties['examples']."""
        llm = StubLLMBackend(
            concepts=[
                {
                    "name": "Observer Pattern",
                    "description": "Pub-sub for objects",
                    "confidence": 0.9,
                },
                {
                    "name": "BaseballReporter",
                    "description": "Concrete observer for game events",
                    "confidence": 0.7,
                    "concept_role": "illustrative",
                    "illustrates": "Observer Pattern",
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        parent = manifest.concepts[0]
        assert parent.name == "Observer Pattern"
        assert "examples" in parent.properties
        assert len(parent.properties["examples"]) == 1
        assert parent.properties["examples"][0]["name"] == "BaseballReporter"

    def test_illustrative_concept_no_standalone_node(self):
        """Illustrative concept with known parent → NOT in manifest.concepts."""
        llm = StubLLMBackend(
            concepts=[
                {
                    "name": "Observer Pattern",
                    "description": "Pub-sub for objects",
                    "confidence": 0.9,
                },
                {
                    "name": "BaseballReporter",
                    "description": "Concrete observer",
                    "confidence": 0.7,
                    "concept_role": "illustrative",
                    "illustrates": "Observer Pattern",
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        concept_names = [c.name for c in manifest.concepts]
        assert "BaseballReporter" not in concept_names
        assert len(manifest.concepts) == 1

    def test_illustrative_without_parent_falls_back_to_node(self):
        """Illustrative with unknown parent → ConceptNode with role in properties."""
        llm = StubLLMBackend(
            concepts=[
                {
                    "name": "OrphanExample",
                    "description": "Example of nonexistent concept",
                    "confidence": 0.6,
                    "concept_role": "illustrative",
                    "illustrates": "Nonexistent Pattern",
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        assert len(manifest.concepts) == 1
        assert manifest.concepts[0].name == "OrphanExample"

    def test_fallback_node_has_concept_role_property(self):
        """Fallback node has properties['concept_role'] == 'illustrative'."""
        llm = StubLLMBackend(
            concepts=[
                {
                    "name": "OrphanExample",
                    "description": "Example",
                    "confidence": 0.6,
                    "concept_role": "illustrative",
                    "illustrates": "Missing Parent",
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        node = manifest.concepts[0]
        assert node.properties.get("concept_role") == "illustrative"

    def test_fallback_node_has_illustrates_property(self):
        """Fallback node has properties['illustrates'] set."""
        llm = StubLLMBackend(
            concepts=[
                {
                    "name": "OrphanExample",
                    "description": "Example",
                    "confidence": 0.6,
                    "concept_role": "illustrative",
                    "illustrates": "Missing Parent",
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        node = manifest.concepts[0]
        assert node.properties.get("illustrates") == "Missing Parent"


# ===========================================================================
# TestExamplesProperty
# ===========================================================================


class TestExamplesProperty:
    """Verify structure and content of properties['examples']."""

    def test_examples_property_populated(self):
        """parent.properties['examples'] is a list of dicts."""
        llm = StubLLMBackend(
            concepts=[
                {"name": "Strategy Pattern", "description": "Encapsulate algorithms"},
                {
                    "name": "SortStrategy",
                    "description": "Concrete sorting strategy",
                    "concept_role": "illustrative",
                    "illustrates": "Strategy Pattern",
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        examples = manifest.concepts[0].properties["examples"]
        assert isinstance(examples, list)
        assert all(isinstance(e, dict) for e in examples)

    def test_example_dict_has_required_fields(self):
        """Each example dict has name, description, source_location, confidence."""
        llm = StubLLMBackend(
            concepts=[
                {"name": "Strategy Pattern", "description": "Encapsulate algorithms"},
                {
                    "name": "SortStrategy",
                    "description": "Sorts things",
                    "confidence": 0.75,
                    "concept_role": "illustrative",
                    "illustrates": "Strategy Pattern",
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        example = manifest.concepts[0].properties["examples"][0]
        assert example["name"] == "SortStrategy"
        assert example["description"] == "Sorts things"
        assert "source_location" in example
        assert example["confidence"] == 0.75

    def test_multiple_examples_on_same_parent(self):
        """Multiple illustrative concepts for same parent all attached."""
        llm = StubLLMBackend(
            concepts=[
                {"name": "Observer Pattern", "description": "Pub-sub"},
                {
                    "name": "BaseballReporter",
                    "description": "Reports hits",
                    "concept_role": "illustrative",
                    "illustrates": "Observer Pattern",
                },
                {
                    "name": "StockTicker",
                    "description": "Displays prices",
                    "concept_role": "illustrative",
                    "illustrates": "Observer Pattern",
                },
                {
                    "name": "WeatherDisplay",
                    "description": "Shows weather",
                    "concept_role": "illustrative",
                    "illustrates": "Observer Pattern",
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        assert len(manifest.concepts) == 1
        examples = manifest.concepts[0].properties["examples"]
        assert len(examples) == 3
        names = {e["name"] for e in examples}
        assert names == {"BaseballReporter", "StockTicker", "WeatherDisplay"}

    def test_examples_property_not_created_when_no_examples(self):
        """Concepts without illustrative children have no 'examples' key."""
        llm = StubLLMBackend(
            concepts=[
                {"name": "Singleton", "description": "Single instance"},
                {"name": "Factory", "description": "Create objects"},
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        for concept in manifest.concepts:
            assert "examples" not in concept.properties


# ===========================================================================
# TestBackwardCompatibility
# ===========================================================================


class TestBackwardCompatibility:
    """Ensure old-format concepts (no concept_role) still work."""

    def test_no_concept_role_defaults_to_generalizable(self):
        """If LLM doesn't return concept_role, concept treated as generalizable."""
        llm = StubLLMBackend(
            concepts=[
                {"name": "Pure Function", "description": "No side effects", "confidence": 0.9},
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="fp")

        assert len(manifest.concepts) == 1
        assert manifest.concepts[0].name == "Pure Function"
        # No concept_role in properties (it's generalizable, the default)
        assert "concept_role" not in manifest.concepts[0].properties

    def test_mixed_old_and_new_format(self):
        """Mix of concepts with and without concept_role."""
        llm = StubLLMBackend(
            concepts=[
                # Old format — no concept_role
                {"name": "Observer Pattern", "description": "Pub-sub"},
                # New format — generalizable explicit
                {
                    "name": "Strategy Pattern",
                    "description": "Algorithms",
                    "concept_role": "generalizable",
                },
                # New format — illustrative
                {
                    "name": "BaseballReporter",
                    "description": "Observer example",
                    "concept_role": "illustrative",
                    "illustrates": "Observer Pattern",
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        assert len(manifest.concepts) == 2
        names = {c.name for c in manifest.concepts}
        assert names == {"Observer Pattern", "Strategy Pattern"}

    def test_empty_illustrates_treated_as_generalizable(self):
        """Illustrative with null/empty illustrates → generalizable (no parent to match)."""
        llm = StubLLMBackend(
            concepts=[
                {
                    "name": "WeirdConcept",
                    "description": "Illustrative but no parent",
                    "concept_role": "illustrative",
                    "illustrates": None,
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        # Falls back to a node (parent=None → no match)
        assert len(manifest.concepts) == 1
        assert manifest.concepts[0].name == "WeirdConcept"
        assert manifest.concepts[0].properties.get("concept_role") == "illustrative"

    def test_unexpected_concept_role_treated_as_generalizable(self):
        """Unexpected concept_role value (e.g. typo) treated as generalizable."""
        llm = StubLLMBackend(
            concepts=[
                {
                    "name": "Some Concept",
                    "description": "Has a typo in role",
                    "concept_role": "illlustrative",  # typo
                    "illustrates": "Something",
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        # Treated as generalizable (else branch), so it becomes a standalone node
        assert len(manifest.concepts) == 1
        assert manifest.concepts[0].name == "Some Concept"
        # No concept_role in properties since it went through the generalizable path
        assert "concept_role" not in manifest.concepts[0].properties

    def test_stub_backend_default_no_args(self):
        """StubLLMBackend() with no args still works (backward compat)."""
        llm = StubLLMBackend()
        assert llm.extract_concepts("text") == []
        assert llm.extract_relations([], "text") == []
        assert llm.extract_rules("text", []) == []
        assert llm.suggest_domain_name("test", "text") == "test"


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_illustrative_concept_case_insensitive_parent_match(self):
        """'observer pattern' matches 'Observer Pattern' (case-insensitive)."""
        llm = StubLLMBackend(
            concepts=[
                {"name": "Observer Pattern", "description": "Pub-sub"},
                {
                    "name": "GameReporter",
                    "description": "Concrete observer",
                    "concept_role": "illustrative",
                    "illustrates": "observer pattern",  # lowercase
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        assert len(manifest.concepts) == 1
        assert "examples" in manifest.concepts[0].properties
        assert manifest.concepts[0].properties["examples"][0]["name"] == "GameReporter"

    def test_all_illustrative_yields_fallback_nodes(self):
        """All illustrative, no parents → all become fallback nodes."""
        llm = StubLLMBackend(
            concepts=[
                {
                    "name": "ExampleA",
                    "description": "Example of X",
                    "concept_role": "illustrative",
                    "illustrates": "X",
                },
                {
                    "name": "ExampleB",
                    "description": "Example of Y",
                    "concept_role": "illustrative",
                    "illustrates": "Y",
                },
            ]
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        assert len(manifest.concepts) == 2
        for c in manifest.concepts:
            assert c.properties.get("concept_role") == "illustrative"

    def test_manifest_relations_only_reference_surviving_concepts(self):
        """Edges referencing filtered-out illustrative concepts are removed."""
        llm = StubLLMBackend(
            concepts=[
                {"name": "Observer Pattern", "description": "Pub-sub"},
                {
                    "name": "BaseballReporter",
                    "description": "Concrete observer",
                    "concept_role": "illustrative",
                    "illustrates": "Observer Pattern",
                },
            ],
            relations=[
                {
                    "source_id": "design:Observer Pattern",
                    "target_id": "design:BaseballReporter",
                    "relation_type": "USES",
                    "confidence": 0.8,
                },
            ],
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        # BaseballReporter was absorbed as example, so the edge target is invalid
        # The edge filter in ingest() checks valid_ids, so it should be dropped
        valid_ids = {c.id for c in manifest.concepts}
        for edge in manifest.edges:
            assert edge.source_id in valid_ids
            assert edge.target_id in valid_ids

    def test_manifest_rules_only_reference_surviving_concepts(self):
        """Rules referencing filtered-out illustrative concepts have IDs removed."""
        llm = StubLLMBackend(
            concepts=[
                {"name": "Observer Pattern", "description": "Pub-sub"},
                {
                    "name": "BaseballReporter",
                    "description": "Concrete observer",
                    "concept_role": "illustrative",
                    "illustrates": "Observer Pattern",
                },
            ],
            rules=[
                {
                    "text": "Use observer for decoupling",
                    "concept_ids": ["design:Observer Pattern", "design:BaseballReporter"],
                    "category": "best_practice",
                    "confidence": 0.9,
                },
            ],
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        # Rule concept_ids should only contain surviving concept IDs
        valid_ids = {c.id for c in manifest.concepts}
        for rule in manifest.rules:
            for cid in rule.concept_ids:
                assert cid in valid_ids

    def test_source_location_preserved_in_examples(self):
        """Example source_location comes from the chunk, not the parent."""
        chunks = [
            Chunk(id="ch1", content="observer stuff", location="Chapter 1"),
            Chunk(id="ch2", content="baseball reporter", location="Chapter 2"),
        ]

        # We need different concepts per chunk call
        class ChunkAwareLLM(StubLLMBackend):
            def __init__(self):
                super().__init__()
                self._call_count = 0

            def extract_concepts(self, text: str, domain_hint: str | None = None) -> list[dict]:
                self._call_count += 1
                if self._call_count == 1:
                    return [{"name": "Observer Pattern", "description": "Pub-sub"}]
                return [
                    {
                        "name": "BaseballReporter",
                        "description": "Concrete observer",
                        "concept_role": "illustrative",
                        "illustrates": "Observer Pattern",
                    }
                ]

        ingestor = SimpleIngestor(ChunkAwareLLM(), chunks=chunks)
        manifest = ingestor.ingest(_make_source(), domain="design")

        example = manifest.concepts[0].properties["examples"][0]
        assert example["source_location"] == "Chapter 2"


# ===========================================================================
# TestIntegrationWithPipeline
# ===========================================================================


class TestIntegrationWithPipeline:
    """Integration tests verifying downstream behavior."""

    def test_illustrative_concepts_not_projected_as_rules(self):
        """Full pipeline: illustrative concepts attached as examples don't generate rules."""
        llm = StubLLMBackend(
            concepts=[
                {"name": "Observer Pattern", "description": "Pub-sub for objects"},
                {
                    "name": "BaseballReporter",
                    "description": "Concrete observer for game events",
                    "concept_role": "illustrative",
                    "illustrates": "Observer Pattern",
                },
            ],
            rules=[
                {
                    "text": "Use the Observer pattern for event-driven decoupling",
                    "concept_ids": ["design:Observer Pattern"],
                    "category": "best_practice",
                    "confidence": 0.9,
                },
            ],
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        # Only generalizable concepts survive
        assert len(manifest.concepts) == 1
        assert manifest.concepts[0].name == "Observer Pattern"

        # Rules only reference the surviving concept
        for rule in manifest.rules:
            assert "design:BaseballReporter" not in rule.concept_ids

    def test_observer_pattern_scenario(self):
        """The BaseballReporter scenario: Observer + BaseballReporter → only Observer rules projected."""
        llm = StubLLMBackend(
            concepts=[
                {
                    "name": "Observer Pattern",
                    "description": "Define a one-to-many dependency between objects",
                    "confidence": 0.95,
                    "concept_role": "generalizable",
                },
                {
                    "name": "Subject Interface",
                    "description": "Interface for objects being observed",
                    "confidence": 0.9,
                    "concept_role": "generalizable",
                },
                {
                    "name": "BaseballReporter",
                    "description": "Concrete observer that reports baseball game events",
                    "confidence": 0.7,
                    "concept_role": "illustrative",
                    "illustrates": "Observer Pattern",
                },
                {
                    "name": "HitEvent",
                    "description": "Event fired when a batter gets a hit",
                    "confidence": 0.6,
                    "concept_role": "illustrative",
                    "illustrates": "Observer Pattern",
                },
                {
                    "name": "GameScoreboard",
                    "description": "Display that updates when game state changes",
                    "confidence": 0.65,
                    "concept_role": "illustrative",
                    "illustrates": "Subject Interface",
                },
            ],
            rules=[
                {
                    "text": "Use Observer pattern when changes to one object require updating others",
                    "concept_ids": ["design:Observer Pattern"],
                    "category": "best_practice",
                    "confidence": 0.9,
                },
            ],
        )
        ingestor = SimpleIngestor(llm)
        manifest = ingestor.ingest(_make_source(), domain="design")

        # Only 2 generalizable concepts survive
        concept_names = {c.name for c in manifest.concepts}
        assert concept_names == {"Observer Pattern", "Subject Interface"}

        # Observer has 2 examples, Subject Interface has 1
        observer = next(c for c in manifest.concepts if c.name == "Observer Pattern")
        subject = next(c for c in manifest.concepts if c.name == "Subject Interface")

        assert len(observer.properties["examples"]) == 2
        example_names = {e["name"] for e in observer.properties["examples"]}
        assert example_names == {"BaseballReporter", "HitEvent"}

        assert len(subject.properties["examples"]) == 1
        assert subject.properties["examples"][0]["name"] == "GameScoreboard"

        # Rules don't reference illustrative concepts
        for rule in manifest.rules:
            for cid in rule.concept_ids:
                assert cid in {c.id for c in manifest.concepts}
