"""Tests for qortex-online concept extraction."""

from __future__ import annotations

import pytest

from qortex.online.extractor import (
    ExtractionResult,
    ExtractionStrategy,
    ExtractedConcept,
    ExtractedRelation,
    LLMExtractor,
    NullExtractor,
    SpaCyExtractor,
    _deduplicate_spans,
    _title_case,
)


# ---------------------------------------------------------------------------
# Data type tests
# ---------------------------------------------------------------------------


class TestExtractionResult:
    def test_empty_by_default(self):
        r = ExtractionResult()
        assert r.empty
        assert r.concepts == []
        assert r.relations == []

    def test_not_empty_with_concepts(self):
        r = ExtractionResult(concepts=[
            ExtractedConcept(name="JWT", description="token format"),
        ])
        assert not r.empty

    def test_frozen(self):
        r = ExtractionResult()
        with pytest.raises(AttributeError):
            r.concepts = []  # type: ignore[misc]

    def test_concept_frozen(self):
        c = ExtractedConcept(name="Auth", description="module", confidence=0.9)
        with pytest.raises(AttributeError):
            c.name = "Other"  # type: ignore[misc]

    def test_relation_frozen(self):
        r = ExtractedRelation(
            source_name="A", target_name="B",
            relation_type="USES", confidence=0.7,
        )
        with pytest.raises(AttributeError):
            r.relation_type = "REQUIRES"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Helper tests
# ---------------------------------------------------------------------------


class TestTitleCase:
    def test_basic(self):
        assert _title_case("hello world") == "Hello World"

    def test_strips_whitespace(self):
        assert _title_case("  foo  bar  ") == "Foo Bar"

    def test_empty(self):
        assert _title_case("") == ""


class TestDeduplicateSpans:
    def test_entities_preferred_over_noun_chunks(self):
        entities = [(0, 5, "Apple", "ORG")]
        noun_chunks = [(0, 5, "Apple")]
        result = _deduplicate_spans(entities, noun_chunks)
        assert len(result) == 1
        assert result[0].confidence == 0.9  # entity confidence

    def test_non_overlapping_chunks_included(self):
        entities = [(0, 5, "Apple", "ORG")]
        noun_chunks = [(10, 20, "the market")]
        result = _deduplicate_spans(entities, noun_chunks)
        assert len(result) == 2

    def test_empty_inputs(self):
        assert _deduplicate_spans([], []) == []

    def test_short_names_filtered(self):
        entities = [(0, 1, "A", "ORG")]
        result = _deduplicate_spans(entities, [])
        assert len(result) == 0


# ---------------------------------------------------------------------------
# NullExtractor tests
# ---------------------------------------------------------------------------


class TestNullExtractor:
    def test_always_empty(self):
        ext = NullExtractor()
        result = ext("The auth module handles JWT tokens.")
        assert result.empty

    def test_conforms_to_protocol(self):
        ext: ExtractionStrategy = NullExtractor()
        assert ext("test").empty


# ---------------------------------------------------------------------------
# SpaCyExtractor tests
# ---------------------------------------------------------------------------


class TestSpaCyExtractor:
    @pytest.fixture()
    def extractor(self):
        """Create SpaCyExtractor, skip if spaCy not installed."""
        ext = SpaCyExtractor()
        if not ext._ensure_loaded():
            pytest.skip("spaCy not installed")
        return ext

    def test_empty_input(self, extractor):
        assert extractor("").empty
        assert extractor("   ").empty

    def test_extracts_named_entities(self, extractor):
        result = extractor("Google announced a new product in New York.")
        names = [c.name.lower() for c in result.concepts]
        # spaCy should find at least Google and New York
        assert any("google" in n for n in names)

    def test_extracts_noun_chunks(self, extractor):
        result = extractor("The authentication module validates user credentials.")
        assert len(result.concepts) >= 1

    def test_deduplicates_overlapping_spans(self, extractor):
        result = extractor("Microsoft released Windows in Seattle.")
        names = [c.name.lower() for c in result.concepts]
        # No exact duplicates
        assert len(names) == len(set(names))

    def test_returns_empty_not_crash_when_no_entities(self, extractor):
        result = extractor("a b c d e f g")
        # May or may not find concepts, but must not crash
        assert isinstance(result, ExtractionResult)

    def test_conforms_to_protocol(self, extractor):
        ext: ExtractionStrategy = extractor
        result = ext("Test sentence.")
        assert isinstance(result, ExtractionResult)

    def test_graceful_without_spacy(self, monkeypatch):
        """If spaCy import fails, returns empty."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "spacy":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        ext = SpaCyExtractor()
        ext._available = None  # reset
        monkeypatch.setattr(builtins, "__import__", mock_import)
        result = ext("Google is great.")
        assert result.empty


# ---------------------------------------------------------------------------
# LLMExtractor tests
# ---------------------------------------------------------------------------


class TestLLMExtractor:
    def test_wraps_backend_concepts(self):
        class MockBackend:
            def extract_concepts(self, text, domain_hint=None):
                return [
                    {"name": "JWT Tokens", "description": "Auth format", "confidence": 0.9},
                    {"name": "Auth Module", "description": "Handles auth", "confidence": 0.85},
                ]

            def extract_relations(self, concepts, text, chunk_location=None):
                return [
                    {"source": "Auth Module", "target": "JWT Tokens",
                     "relation_type": "USES", "confidence": 0.8},
                ]

        ext = LLMExtractor(MockBackend())
        result = ext("The auth module handles JWT tokens.")
        assert len(result.concepts) == 2
        assert result.concepts[0].name == "JWT Tokens"
        assert len(result.relations) == 1
        assert result.relations[0].relation_type == "USES"

    def test_empty_text_returns_empty(self):
        ext = LLMExtractor(None)
        assert ext("").empty
        assert ext("  ").empty

    def test_never_crashes_on_backend_error(self):
        class BrokenBackend:
            def extract_concepts(self, text, domain_hint=None):
                raise RuntimeError("backend exploded")

        ext = LLMExtractor(BrokenBackend())
        result = ext("Test text.")
        assert result.empty

    def test_empty_concepts_returns_empty(self):
        class EmptyBackend:
            def extract_concepts(self, text, domain_hint=None):
                return []

        ext = LLMExtractor(EmptyBackend())
        result = ext("Test text.")
        assert result.empty

    def test_filters_nameless_concepts(self):
        class PartialBackend:
            def extract_concepts(self, text, domain_hint=None):
                return [
                    {"name": "Good", "description": "ok"},
                    {"description": "no name"},
                    {"name": "", "description": "empty name"},
                ]

            def extract_relations(self, concepts, text, chunk_location=None):
                return []

        ext = LLMExtractor(PartialBackend())
        result = ext("Test.")
        assert len(result.concepts) == 1
        assert result.concepts[0].name == "Good"


# ---------------------------------------------------------------------------
# Extraction strategy injection tests
# ---------------------------------------------------------------------------


class TestExtractionStrategyInjection:
    def test_custom_strategy_is_used(self):
        from qortex.mcp.server import set_extraction_strategy

        call_count = 0

        class CountingExtractor:
            def __call__(self, text, domain=""):
                nonlocal call_count
                call_count += 1
                return ExtractionResult(concepts=[
                    ExtractedConcept(name="Test", description="test"),
                ])

        set_extraction_strategy(CountingExtractor(), name="test")

        from qortex.mcp.server import _get_extractor

        ext = _get_extractor()
        result = ext("Hello.")
        assert call_count == 1
        assert len(result.concepts) == 1

        # Reset
        set_extraction_strategy(None)

    def test_null_reset(self):
        from qortex.mcp.server import set_extraction_strategy

        set_extraction_strategy(NullExtractor(), name="none")
        from qortex.mcp.server import _get_extractor

        ext = _get_extractor()
        assert ext("test").empty

        # Reset
        set_extraction_strategy(None)
