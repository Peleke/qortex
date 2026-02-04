"""Tests for the strategy system."""

import pytest

from qortex_ingest import Pipeline, Source, ingest
from qortex_ingest.llm import get_llm_backend
from qortex_ingest.strategies import (
    available_input_strategies,
    available_output_strategies,
    get_input_strategy,
    get_output_strategy,
)
from qortex_ingest.strategies.input import (
    MarkdownInputStrategy,
    TextInputStrategy,
)
from qortex_ingest.strategies.output import (
    ExtractionResult,
    JSONOutputStrategy,
    ManifestOutputStrategy,
    YAMLOutputStrategy,
)


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_MARKDOWN = """# Functional Programming

Functional programming emphasizes immutability and pure functions.

## Pure Functions

A pure function always returns the same output for the same input.

- Avoid side effects
- Don't modify arguments
- Return new values instead of mutating

## Immutability

Data should not be changed after creation.

1. Use const instead of let
2. Prefer spread operator over mutation
3. Use immutable data structures
"""

SAMPLE_TEXT = """
Pure functions are a core concept in functional programming.
They always return the same output for the same input and have no side effects.

Avoid mutable state in your code. Instead of modifying objects,
return new objects with the desired changes.

Key principles:
- Always prefer const over let
- Never mutate function arguments
- Use map/filter/reduce over for loops
"""


# =============================================================================
# Input Strategy Tests
# =============================================================================


class TestInputStrategies:
    """Tests for input strategies."""

    def test_available_strategies(self):
        """Should have text, markdown, pdf strategies."""
        strategies = available_input_strategies()
        assert "text" in strategies
        assert "markdown" in strategies
        assert "pdf" in strategies

    def test_get_text_strategy(self):
        """Should get text strategy by name."""
        strategy = get_input_strategy("text")
        assert strategy.name == "text"
        assert isinstance(strategy, TextInputStrategy)

    def test_get_markdown_strategy(self):
        """Should get markdown strategy by name."""
        strategy = get_input_strategy("markdown")
        assert strategy.name == "markdown"
        assert isinstance(strategy, MarkdownInputStrategy)

    def test_auto_detect_markdown(self):
        """Should auto-detect markdown from source."""
        source = Source(
            raw_content=SAMPLE_MARKDOWN,
            source_type="markdown",
            name="test.md",
        )
        strategy = get_input_strategy(source=source)
        assert strategy.name == "markdown"

    def test_auto_detect_text(self):
        """Should auto-detect text from source."""
        source = Source(
            raw_content=SAMPLE_TEXT,
            source_type="text",
            name="test.txt",
        )
        strategy = get_input_strategy(source=source)
        assert strategy.name == "text"

    def test_text_chunking(self):
        """Text strategy should chunk by size."""
        strategy = TextInputStrategy(chunk_size=200, chunk_overlap=50)
        source = Source(raw_content=SAMPLE_TEXT, name="test")

        chunks = strategy.chunk(source)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 250  # Approximate

    def test_markdown_chunking(self):
        """Markdown strategy should chunk by headings."""
        strategy = MarkdownInputStrategy()
        source = Source(raw_content=SAMPLE_MARKDOWN, source_type="markdown", name="test")

        chunks = strategy.chunk(source)

        # Should have chunks for main heading and subheadings
        assert len(chunks) >= 3

        # Check locations
        locations = [c.location for c in chunks]
        assert any("Functional Programming" in loc for loc in locations)
        assert any("Pure Functions" in loc for loc in locations)

    def test_unknown_strategy_raises(self):
        """Should raise for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown input strategy"):
            get_input_strategy("nonexistent")


# =============================================================================
# Output Strategy Tests
# =============================================================================


class TestOutputStrategies:
    """Tests for output strategies."""

    @pytest.fixture
    def sample_result(self):
        """Sample extraction result for testing."""
        from qortex.core.models import (
            ConceptEdge,
            ConceptNode,
            ExplicitRule,
            RelationType,
            SourceMetadata,
        )

        return ExtractionResult(
            source=SourceMetadata(
                id="test:source",
                name="Test Source",
                source_type="text",
                path_or_url="test.txt",
            ),
            domain="test_domain",
            concepts=[
                ConceptNode(
                    id="test:pure_function",
                    name="Pure Function",
                    description="A function with no side effects",
                    domain="test_domain",
                    source_id="test:source",
                ),
                ConceptNode(
                    id="test:immutability",
                    name="Immutability",
                    description="Data that cannot be changed",
                    domain="test_domain",
                    source_id="test:source",
                ),
            ],
            edges=[
                ConceptEdge(
                    source_id="test:pure_function",
                    target_id="test:immutability",
                    relation_type=RelationType.REQUIRES,
                ),
            ],
            rules=[
                ExplicitRule(
                    id="test:rule:0",
                    text="Avoid side effects in functions",
                    domain="test_domain",
                    source_id="test:source",
                    category="architectural",
                ),
            ],
        )

    def test_available_strategies(self):
        """Should have manifest, json, yaml strategies."""
        strategies = available_output_strategies()
        assert "manifest" in strategies
        assert "json" in strategies
        assert "yaml" in strategies

    def test_manifest_output(self, sample_result):
        """Manifest strategy should return IngestionManifest."""
        strategy = ManifestOutputStrategy()
        result = strategy.output(sample_result)

        assert result.domain == "test_domain"
        assert len(result.concepts) == 2
        assert len(result.edges) == 1
        assert len(result.rules) == 1

    def test_json_output(self, sample_result):
        """JSON strategy should return valid JSON string."""
        import json

        strategy = JSONOutputStrategy()
        result = strategy.output(sample_result)

        # Should be valid JSON
        data = json.loads(result)
        assert data["domain"] == "test_domain"
        assert len(data["concepts"]) == 2
        assert len(data["rules"]) == 1

    def test_yaml_output(self, sample_result):
        """YAML strategy should return valid YAML string."""
        import yaml

        strategy = YAMLOutputStrategy()
        result = strategy.output(sample_result)

        # Should be valid YAML
        data = yaml.safe_load(result)
        assert data["domain"] == "test_domain"
        assert len(data["concepts"]) == 2

    def test_yaml_rules_only(self, sample_result):
        """YAML strategy with rules_only should exclude concepts."""
        import yaml

        strategy = YAMLOutputStrategy(rules_only=True)
        result = strategy.output(sample_result)

        data = yaml.safe_load(result)
        assert "rules" in data
        assert "concepts" not in data

    def test_unknown_strategy_raises(self):
        """Should raise for unknown strategy."""
        with pytest.raises(ValueError, match="Unknown output strategy"):
            get_output_strategy("nonexistent")


# =============================================================================
# Pipeline Tests
# =============================================================================


class TestPipeline:
    """Tests for the Pipeline class."""

    def test_pipeline_with_text(self):
        """Pipeline should process text source."""
        llm = get_llm_backend("keyword")
        pipeline = Pipeline(
            input_strategy=get_input_strategy("text"),
            llm_backend=llm,
            output_strategy=get_output_strategy("manifest"),
        )

        source = Source(raw_content=SAMPLE_TEXT, name="test")
        result = pipeline.process(source, domain="fp_test")

        assert result.domain == "fp_test"
        assert len(result.concepts) > 0
        assert len(result.rules) > 0

    def test_pipeline_with_markdown(self):
        """Pipeline should process markdown source."""
        llm = get_llm_backend("keyword")
        pipeline = Pipeline(
            input_strategy=get_input_strategy("markdown"),
            llm_backend=llm,
            output_strategy=get_output_strategy("json"),
        )

        source = Source(raw_content=SAMPLE_MARKDOWN, source_type="markdown", name="test")
        result = pipeline.process(source, domain="fp_test")

        # Should be valid JSON
        import json
        data = json.loads(result)
        assert data["domain"] == "fp_test"

    def test_pipeline_auto(self):
        """Pipeline.auto should auto-detect input strategy."""
        source = Source(raw_content=SAMPLE_MARKDOWN, source_type="markdown", name="test")
        llm = get_llm_backend("keyword")

        pipeline = Pipeline.auto(source, llm, output="yaml")
        result = pipeline.process(source, domain="auto_test")

        import yaml
        data = yaml.safe_load(result)
        assert data["domain"] == "auto_test"

    def test_ingest_convenience(self):
        """ingest() convenience function should work."""
        source = Source(raw_content=SAMPLE_TEXT, name="test")
        llm = get_llm_backend("keyword")

        result = ingest(source, llm, domain="convenience_test", output="manifest")

        assert result.domain == "convenience_test"
        assert len(result.concepts) > 0

    def test_pipeline_deduplicates_concepts(self):
        """Pipeline should deduplicate concepts by name."""
        # Text with repeated concepts
        text = """
        Pure functions are important. Pure functions have no side effects.
        Pure functions always return same output. Pure functions are testable.
        """

        llm = get_llm_backend("keyword")
        pipeline = Pipeline(
            input_strategy=get_input_strategy("text", chunk_size=50),
            llm_backend=llm,
            output_strategy=get_output_strategy("manifest"),
        )

        source = Source(raw_content=text, name="test")
        result = pipeline.process(source)

        # "Pure" or similar should only appear once
        names = [c.name.lower() for c in result.concepts]
        assert len(names) == len(set(names))  # No duplicates


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests across the strategy system."""

    def test_full_pipeline_text_to_yaml(self):
        """Full pipeline: text -> extract -> yaml."""
        import yaml

        source = Source(raw_content=SAMPLE_TEXT, name="fp_guide")
        llm = get_llm_backend()  # Auto-select

        result = ingest(source, llm, output="yaml")

        # Verify output
        data = yaml.safe_load(result)
        assert "domain" in data
        assert "rules" in data

    def test_full_pipeline_markdown_to_json(self):
        """Full pipeline: markdown -> extract -> json."""
        import json

        source = Source(raw_content=SAMPLE_MARKDOWN, source_type="markdown", name="fp_guide")
        llm = get_llm_backend()

        result = ingest(source, llm, output="json")

        data = json.loads(result)
        assert "concepts" in data
        assert "edges" in data
        assert "rules" in data
