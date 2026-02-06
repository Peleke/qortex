"""Tests for core data models."""

import json
from dataclasses import asdict

from qortex.core.models import (
    CodeExample,
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


# =============================================================================
# CodeExample Tests
# =============================================================================


class TestCodeExample:
    """Exhaustive tests for CodeExample model."""

    def test_code_example_creation_minimal(self):
        """CodeExample can be created with minimal required fields."""
        example = CodeExample(
            id="test:example:0",
            code="def hello(): pass",
            language="python",
        )

        assert example.id == "test:example:0"
        assert example.code == "def hello(): pass"
        assert example.language == "python"
        assert example.description is None
        assert example.source_location is None
        assert example.concept_ids == []
        assert example.rule_ids == []
        assert example.tags == []
        assert example.is_antipattern is False
        assert example.properties == {}

    def test_code_example_creation_full(self):
        """CodeExample can be created with all fields."""
        example = CodeExample(
            id="patterns:example:iterator",
            code="class Iterator:\n    def __iter__(self): return self\n    def __next__(self): ...",
            language="python",
            description="Shows the iterator protocol implementation",
            source_location="ch11:p42",
            concept_ids=["patterns:Iterator", "patterns:Protocol"],
            rule_ids=["patterns:rule:0"],
            tags=["iterator", "design-pattern", "protocol"],
            is_antipattern=False,
            properties={"complexity": "low", "tested": True},
        )

        assert example.id == "patterns:example:iterator"
        assert "class Iterator" in example.code
        assert example.language == "python"
        assert example.description == "Shows the iterator protocol implementation"
        assert example.source_location == "ch11:p42"
        assert "patterns:Iterator" in example.concept_ids
        assert "patterns:rule:0" in example.rule_ids
        assert "iterator" in example.tags
        assert example.is_antipattern is False
        assert example.properties["complexity"] == "low"

    def test_code_example_antipattern(self):
        """CodeExample can represent an antipattern."""
        example = CodeExample(
            id="test:antipattern:0",
            code="# Bad: mutable default argument\ndef bad(items=[]):\n    items.append(1)",
            language="python",
            description="Mutable default argument causes unexpected behavior",
            is_antipattern=True,
            tags=["antipattern", "mutable-default"],
        )

        assert example.is_antipattern is True
        assert "antipattern" in example.tags

    def test_code_example_serialization_to_dict(self):
        """CodeExample serializes to dict (SQLA compatible)."""
        example = CodeExample(
            id="test:example:0",
            code="print('hello')",
            language="python",
            description="Simple print",
            concept_ids=["test:concept"],
            tags=["simple"],
        )

        data = asdict(example)

        assert data["id"] == "test:example:0"
        assert data["code"] == "print('hello')"
        assert data["language"] == "python"
        assert data["description"] == "Simple print"
        assert data["concept_ids"] == ["test:concept"]
        assert data["tags"] == ["simple"]
        assert data["is_antipattern"] is False

    def test_code_example_serialization_to_json(self):
        """CodeExample serializes to JSON and back."""
        example = CodeExample(
            id="test:example:json",
            code="const x = 1;",
            language="javascript",
            description="Variable declaration",
            tags=["js", "const"],
        )

        # Serialize
        data = asdict(example)
        json_str = json.dumps(data)

        # Deserialize
        loaded = json.loads(json_str)
        restored = CodeExample(**loaded)

        assert restored.id == example.id
        assert restored.code == example.code
        assert restored.language == example.language
        assert restored.tags == example.tags

    def test_code_example_sqla_compatible_structure(self):
        """CodeExample dict structure matches SQLA model expectations.

        This ensures dict can be passed directly to SQLA CodeExample(**dict).
        """
        example = CodeExample(
            id="test:sqla:0",
            code="SELECT * FROM users",
            language="sql",
            description="Query all users",
            source_location="docs/queries.md",
            concept_ids=["db:users"],
            rule_ids=["db:rule:select"],
            tags=["sql", "query"],
            is_antipattern=False,
            properties={"reviewed": True},
        )

        data = asdict(example)

        # SQLA model expected columns
        sqla_columns = {
            "id",
            "code",
            "language",
            "description",
            "source_location",
            "is_antipattern",
            "properties",
        }

        # All SQLA columns present
        for col in sqla_columns:
            assert col in data, f"Missing SQLA column: {col}"

        # M2M fields present (handled separately in SQLA)
        assert "concept_ids" in data
        assert "rule_ids" in data
        assert "tags" in data

    def test_code_example_multiline_code(self):
        """CodeExample preserves multiline code formatting."""
        code = """class Visitor:
    def visit_element_a(self, element):
        pass

    def visit_element_b(self, element):
        pass"""

        example = CodeExample(
            id="test:multiline",
            code=code,
            language="python",
        )

        assert example.code == code
        assert "\n" in example.code
        assert "def visit_element_a" in example.code

    def test_code_example_various_languages(self):
        """CodeExample supports various programming languages."""
        languages = [
            ("python", "def foo(): pass"),
            ("javascript", "const foo = () => {}"),
            ("typescript", "const foo: () => void = () => {}"),
            ("java", "public void foo() {}"),
            ("go", "func foo() {}"),
            ("rust", "fn foo() {}"),
            ("sql", "SELECT 1"),
            ("cypher", "MATCH (n) RETURN n"),
            ("unknown", "some code"),
        ]

        for lang, code in languages:
            example = CodeExample(id=f"test:{lang}", code=code, language=lang)
            assert example.language == lang
            assert example.code == code

    def test_code_example_empty_lists_default(self):
        """CodeExample defaults to empty lists, not None."""
        example = CodeExample(
            id="test:defaults",
            code="x",
            language="text",
        )

        # Should be empty lists, not None
        assert example.concept_ids == []
        assert example.rule_ids == []
        assert example.tags == []
        assert example.properties == {}

        # Should be mutable (not shared)
        example.concept_ids.append("test:concept")
        example2 = CodeExample(id="test:defaults2", code="y", language="text")
        assert example2.concept_ids == []  # Not affected

    def test_code_example_in_manifest(self):
        """CodeExample integrates with IngestionManifest."""
        source = SourceMetadata(
            id="test_source",
            name="Test",
            source_type="text",
            path_or_url="/test.txt",
        )

        example = CodeExample(
            id="test:example:0",
            code="print(1)",
            language="python",
        )

        manifest = IngestionManifest(
            source=source,
            domain="test",
            concepts=[],
            edges=[],
            rules=[],
            examples=[example],
        )

        assert len(manifest.examples) == 1
        assert manifest.examples[0].code == "print(1)"

    def test_code_example_manifest_serialization(self):
        """IngestionManifest with examples serializes correctly."""
        source = SourceMetadata(
            id="test_source",
            name="Test",
            source_type="text",
            path_or_url="/test.txt",
        )

        example = CodeExample(
            id="test:example:0",
            code="x = 1",
            language="python",
            concept_ids=["test:concept"],
        )

        manifest = IngestionManifest(
            source=source,
            domain="test",
            concepts=[],
            edges=[],
            rules=[],
            examples=[example],
        )

        data = asdict(manifest)

        assert "examples" in data
        assert len(data["examples"]) == 1
        assert data["examples"][0]["code"] == "x = 1"
        assert data["examples"][0]["concept_ids"] == ["test:concept"]
