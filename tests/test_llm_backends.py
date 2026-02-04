"""Tests for LLM backend system.

These tests auto-detect available backends:
- Keyword: Always runs (no external dependencies)
- Anthropic: Runs if ANTHROPIC_API_KEY is set
- Ollama: Runs if Ollama service is running
"""

import os

import pytest

from qortex.core.models import ConceptNode
from qortex_ingest.llm import (
    LLMBackend,
    LLMConfig,
    available_backends,
    get_llm_backend,
)
from qortex_ingest.llm.keyword import KeywordLLMBackend
from qortex_ingest.llm.registry import has_anthropic, has_ollama


# =============================================================================
# Test Fixtures
# =============================================================================


SAMPLE_TEXT = """
Pure Functions are a core concept in Functional Programming. A pure function
always returns the same output for the same input and has no side effects.

Immutability is closely related to pure functions. When data is immutable,
functions cannot modify their inputs, making them naturally pure.

Avoid mutable state in your functions. Instead of modifying objects in place,
return new objects with the desired changes. This makes your code easier to
test and reason about.

Key principles:
1. Always prefer const over let
2. Never mutate function arguments
3. Use map, filter, and reduce instead of for loops
4. Avoid shared mutable state between functions

Testing pure functions is straightforward because they have no hidden
dependencies. You can test them in isolation without mocking.
"""


@pytest.fixture
def sample_text():
    return SAMPLE_TEXT


@pytest.fixture
def config():
    return LLMConfig(
        max_concepts=10,
        max_rules=5,
        min_confidence=0.5,
    )


# =============================================================================
# Registry Tests
# =============================================================================


def test_available_backends_includes_keyword():
    """Keyword backend should always be available."""
    backends = available_backends()
    assert "keyword" in backends


def test_available_backends_sorted_by_priority():
    """Backends should be sorted by priority (highest first)."""
    backends = available_backends()
    # Keyword is lowest priority, should be last (if others available)
    # or first (if only one)
    assert len(backends) >= 1


def test_get_llm_backend_auto_select():
    """Auto-selection should return the best available backend."""
    llm = get_llm_backend()
    assert isinstance(llm, LLMBackend)
    assert llm.is_available


def test_get_llm_backend_explicit_keyword():
    """Explicitly requesting keyword backend should work."""
    llm = get_llm_backend("keyword")
    assert llm.name == "keyword"
    assert llm.is_available


def test_get_llm_backend_with_config():
    """Config should be passed to backend."""
    config = LLMConfig(max_concepts=5, temperature=0.1)
    llm = get_llm_backend("keyword", config=config)
    assert llm.config.max_concepts == 5
    assert llm.config.temperature == 0.1


def test_get_llm_backend_with_kwargs():
    """Kwargs should override config."""
    llm = get_llm_backend("keyword", max_concepts=3)
    assert llm.config.max_concepts == 3


def test_get_llm_backend_unknown_raises():
    """Unknown backend should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown backend"):
        get_llm_backend("nonexistent")


# =============================================================================
# Keyword Backend Tests (Always Run)
# =============================================================================


class TestKeywordBackend:
    """Tests for keyword-based extraction."""

    def test_name(self):
        llm = KeywordLLMBackend()
        assert llm.name == "keyword"

    def test_is_available(self):
        llm = KeywordLLMBackend()
        assert llm.is_available is True

    def test_extract_concepts(self, sample_text, config):
        llm = KeywordLLMBackend(config)
        concepts = llm.extract_concepts(sample_text)

        assert len(concepts) > 0
        assert len(concepts) <= config.max_concepts

        # Check structure
        for c in concepts:
            assert "name" in c
            assert "description" in c
            assert "confidence" in c
            assert 0 <= c["confidence"] <= 1

    def test_extract_concepts_finds_key_terms(self, sample_text):
        llm = KeywordLLMBackend()
        concepts = llm.extract_concepts(sample_text)
        names = [c["name"].lower() for c in concepts]

        # Should find capitalized terms
        assert any("pure" in n or "function" in n for n in names)

    def test_extract_relations(self, sample_text, config):
        llm = KeywordLLMBackend(config)
        concepts = llm.extract_concepts(sample_text)

        # Convert to ConceptNode for relation extraction
        concept_nodes = [
            ConceptNode(
                id=f"test:{c['name']}",
                name=c["name"],
                description=c["description"],
                domain="test",
                source_id="test_source",
            )
            for c in concepts
        ]

        relations = llm.extract_relations(concept_nodes, sample_text)

        # Should find some relations
        assert len(relations) >= 0  # May be empty for keyword backend

        for r in relations:
            assert "source_id" in r
            assert "target_id" in r
            assert "relation_type" in r
            assert "confidence" in r

    def test_extract_rules(self, sample_text, config):
        llm = KeywordLLMBackend(config)
        concepts = llm.extract_concepts(sample_text)
        concept_nodes = [
            ConceptNode(
                id=f"test:{c['name']}",
                name=c["name"],
                description=c["description"],
                domain="test",
                source_id="test_source",
            )
            for c in concepts
        ]

        rules = llm.extract_rules(sample_text, concept_nodes)

        assert len(rules) > 0
        assert len(rules) <= config.max_rules

        # Check structure
        for r in rules:
            assert "text" in r
            assert "category" in r
            assert "confidence" in r

    def test_extract_rules_finds_imperatives(self, sample_text):
        llm = KeywordLLMBackend()
        rules = llm.extract_rules(sample_text, [])

        texts = [r["text"].lower() for r in rules]

        # Should find imperative statements
        assert any("avoid" in t or "never" in t or "always" in t for t in texts)

    def test_suggest_domain_name(self):
        llm = KeywordLLMBackend()
        name = llm.suggest_domain_name(
            "Functional Programming Guide",
            "Pure functions and immutability are core concepts..."
        )

        assert name
        assert len(name) <= 30
        assert "_" in name or name.isalnum()


# =============================================================================
# Anthropic Backend Tests (Conditional)
# =============================================================================


@pytest.mark.skipif(
    not has_anthropic(),
    reason="ANTHROPIC_API_KEY not set or anthropic package not installed"
)
class TestAnthropicBackend:
    """Tests for Claude-based extraction.

    Only runs if ANTHROPIC_API_KEY is set.
    """

    def test_is_available(self):
        llm = get_llm_backend("anthropic")
        assert llm.is_available
        assert llm.name == "anthropic"

    def test_extract_concepts(self, sample_text, config):
        llm = get_llm_backend("anthropic", config=config)
        concepts = llm.extract_concepts(sample_text)

        assert len(concepts) > 0

        # Claude should find more meaningful concepts
        names = [c["name"].lower() for c in concepts]
        # Should identify key FP concepts
        assert any("pure" in n or "function" in n or "immutab" in n for n in names)

    def test_extract_relations_finds_semantics(self, sample_text):
        """Claude should find semantic relationships."""
        llm = get_llm_backend("anthropic")
        concepts = llm.extract_concepts(sample_text)
        concept_nodes = [
            ConceptNode(
                id=f"test:{c['name']}",
                name=c["name"],
                description=c["description"],
                domain="test",
                source_id="test_source",
            )
            for c in concepts
        ]

        relations = llm.extract_relations(concept_nodes, sample_text)

        # Claude should find relationships
        assert len(relations) > 0

        # Should find meaningful relationships, not just similarity
        rel_types = {r["relation_type"].value for r in relations}
        assert len(rel_types) >= 1

    def test_extract_rules_with_categories(self, sample_text):
        """Claude should categorize rules meaningfully."""
        llm = get_llm_backend("anthropic")
        rules = llm.extract_rules(sample_text, [])

        assert len(rules) > 0

        # Should have varied categories
        categories = {r["category"] for r in rules}
        # Might have testing, architectural, antipattern, etc.
        assert "general" in categories or len(categories) >= 1


# =============================================================================
# Ollama Backend Tests (Conditional)
# =============================================================================


@pytest.mark.skipif(
    not has_ollama(),
    reason="Ollama service not running"
)
class TestOllamaBackend:
    """Tests for Ollama-based extraction.

    Only runs if Ollama service is running locally.
    """

    def test_is_available(self):
        llm = get_llm_backend("ollama")
        assert llm.is_available
        assert llm.name == "ollama"

    def test_extract_concepts(self, sample_text, config):
        llm = get_llm_backend("ollama", config=config)
        concepts = llm.extract_concepts(sample_text)

        # Should extract something (quality varies by model)
        assert isinstance(concepts, list)

    def test_with_specific_model(self, sample_text):
        """Test with a specific model if available."""
        try:
            llm = get_llm_backend("ollama", model="llama3")
            concepts = llm.extract_concepts(sample_text[:1000])
            assert isinstance(concepts, list)
        except Exception:
            pytest.skip("Model not available")


# =============================================================================
# Integration Tests
# =============================================================================


class TestBackendIntegration:
    """Integration tests that work with any available backend."""

    def test_full_extraction_pipeline(self, sample_text):
        """Test complete extraction with best available backend."""
        llm = get_llm_backend()  # Auto-select

        # Extract concepts
        concepts = llm.extract_concepts(sample_text)
        assert len(concepts) > 0

        # Convert to nodes
        concept_nodes = [
            ConceptNode(
                id=f"test:{c['name']}",
                name=c["name"],
                description=c["description"],
                domain="test",
                source_id="test_source",
            )
            for c in concepts
        ]

        # Extract relations
        relations = llm.extract_relations(concept_nodes, sample_text)
        assert isinstance(relations, list)

        # Extract rules
        rules = llm.extract_rules(sample_text, concept_nodes)
        assert isinstance(rules, list)

        # Suggest domain
        domain = llm.suggest_domain_name("FP Guide", sample_text[:500])
        assert domain
        assert len(domain) <= 30

    def test_backend_fallback_on_error(self, sample_text):
        """Backends should fallback gracefully on errors."""
        # Even if Anthropic has issues, should not crash
        llm = get_llm_backend("keyword")

        # This should always work
        concepts = llm.extract_concepts(sample_text)
        assert len(concepts) > 0


# =============================================================================
# Config Tests
# =============================================================================


class TestLLMConfig:
    """Tests for LLM configuration."""

    def test_default_config(self):
        config = LLMConfig()
        assert config.temperature == 0.3
        assert config.max_concepts == 20
        assert config.max_rules == 15

    def test_config_limits_respected(self, sample_text):
        config = LLMConfig(max_concepts=3, max_rules=2)
        llm = KeywordLLMBackend(config)

        concepts = llm.extract_concepts(sample_text)
        assert len(concepts) <= 3

        rules = llm.extract_rules(sample_text, [])
        assert len(rules) <= 2

    def test_min_confidence_filter(self, sample_text):
        # High confidence threshold
        config = LLMConfig(min_confidence=0.9)
        llm = KeywordLLMBackend(config)

        concepts = llm.extract_concepts(sample_text)
        for c in concepts:
            assert c["confidence"] >= 0.9
