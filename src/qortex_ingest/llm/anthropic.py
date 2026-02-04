"""Anthropic (Claude) LLM backend for high-quality extraction."""

from __future__ import annotations

import json
import os
import re
from typing import Any

from qortex.core.models import ConceptNode, RelationType

from .base import BaseLLMBackend, LLMConfig

# Lazy import - only load anthropic if actually used
_anthropic_client = None


def _get_anthropic_client(api_key: str | None = None):
    """Lazy-load anthropic client."""
    global _anthropic_client
    if _anthropic_client is None:
        try:
            import anthropic
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if key:
                _anthropic_client = anthropic.Anthropic(api_key=key)
        except ImportError:
            pass
    return _anthropic_client


class AnthropicLLMBackend(BaseLLMBackend):
    """Claude-powered extraction for high-quality results.

    Uses Claude to:
    - Extract concepts with rich descriptions
    - Identify semantic relationships between concepts
    - Extract actionable rules with proper categorization
    """

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    # Extraction prompts
    CONCEPT_EXTRACTION_PROMPT = """Extract key concepts from the following text. For each concept, provide:
- name: A clear, concise name
- description: A brief explanation of what this concept means in context
- confidence: How confident you are this is a meaningful concept (0.0-1.0)

Focus on:
- Technical terms and domain-specific vocabulary
- Named patterns, principles, or methodologies
- Important abstractions or ideas

Return JSON array:
[{"name": "...", "description": "...", "confidence": 0.8}, ...]

Text to analyze:
{text}

{domain_hint}"""

    RELATION_EXTRACTION_PROMPT = """Given these concepts and the source text, identify relationships between them.

Concepts:
{concepts}

Relationship types to look for:
- contradicts: A and B are mutually exclusive or opposing
- requires: A depends on or needs B
- refines: A is a more specific form of B
- implements: A is a concrete realization of B
- similar_to: A and B are related or analogous
- part_of: A is a component of B
- uses: A utilizes or leverages B
- alternative_to: A can substitute for B

Return JSON array:
[{"source": "concept_name", "target": "concept_name", "relation": "relation_type", "confidence": 0.7}, ...]

Source text:
{text}"""

    RULE_EXTRACTION_PROMPT = """Extract actionable rules, guidelines, or best practices from this text.

For each rule:
- text: The rule as a clear, actionable statement
- concepts: List of concept names this rule relates to
- category: One of [architectural, testing, security, performance, antipattern, general]
- confidence: How explicit this rule is in the text (0.0-1.0)

Focus on:
- Explicit recommendations ("Always...", "Never...", "Prefer...")
- Implicit best practices
- Warnings or antipatterns

Known concepts in this text:
{concepts}

Return JSON array:
[{"text": "...", "concepts": ["...", "..."], "category": "...", "confidence": 0.8}, ...]

Text:
{text}"""

    DOMAIN_SUGGESTION_PROMPT = """Suggest a short, snake_case domain name for a knowledge base containing this content.

Source name: {source_name}

Sample content:
{sample_text}

Return only the domain name (e.g., "functional_programming", "api_design", "testing_patterns").
No explanation, just the name."""

    def __init__(self, config: LLMConfig | None = None):
        super().__init__(config)
        self._client = None

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def is_available(self) -> bool:
        """Check if Anthropic API is available."""
        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return False
        try:
            import anthropic
            return True
        except ImportError:
            return False

    @property
    def client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            self._client = _get_anthropic_client(self.config.api_key)
        return self._client

    def _call_claude(self, prompt: str, expect_json: bool = True) -> str | Any:
        """Make a Claude API call."""
        if not self.client:
            raise RuntimeError("Anthropic client not available")

        model = self.config.model or self.DEFAULT_MODEL

        response = self.client.messages.create(
            model=model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text

        if expect_json:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if json_match:
                text = json_match.group(1)

            # Try to find JSON array or object
            json_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', text)
            if json_match:
                text = json_match.group(1)

            return json.loads(text)

        return text.strip()

    def extract_concepts(self, text: str, domain_hint: str | None = None) -> list[dict]:
        """Extract concepts using Claude."""
        hint_text = f"Domain hint: {domain_hint}" if domain_hint else ""

        prompt = self.CONCEPT_EXTRACTION_PROMPT.format(
            text=text[:8000],  # Limit text size
            domain_hint=hint_text,
        )

        try:
            concepts = self._call_claude(prompt, expect_json=True)

            # Validate and normalize
            result = []
            for c in concepts:
                if isinstance(c, dict) and "name" in c:
                    result.append({
                        "name": c["name"],
                        "description": c.get("description", ""),
                        "confidence": float(c.get("confidence", 0.8)),
                    })

            # Filter by confidence
            result = [c for c in result if c["confidence"] >= self.config.min_confidence]

            return result[:self.config.max_concepts]

        except Exception as e:
            # Fallback to keyword extraction on error
            from .keyword import KeywordLLMBackend
            fallback = KeywordLLMBackend(self.config)
            return fallback.extract_concepts(text, domain_hint)

    def extract_relations(
        self,
        concepts: list[ConceptNode],
        text: str,
    ) -> list[dict]:
        """Extract relations using Claude."""
        concept_list = "\n".join(f"- {c.name}: {c.description}" for c in concepts)

        prompt = self.RELATION_EXTRACTION_PROMPT.format(
            concepts=concept_list,
            text=text[:8000],
        )

        try:
            relations = self._call_claude(prompt, expect_json=True)

            # Map concept names to IDs
            name_to_id = {c.name.lower(): c.id for c in concepts}

            # Validate and normalize
            result = []
            for r in relations:
                if isinstance(r, dict) and "source" in r and "target" in r:
                    source_id = name_to_id.get(r["source"].lower())
                    target_id = name_to_id.get(r["target"].lower())

                    if source_id and target_id:
                        try:
                            rel_type = RelationType(r.get("relation", "similar_to").lower())
                        except ValueError:
                            rel_type = RelationType.SIMILAR_TO

                        result.append({
                            "source_id": source_id,
                            "target_id": target_id,
                            "relation_type": rel_type,
                            "confidence": float(r.get("confidence", 0.7)),
                        })

            return result

        except Exception as e:
            from .keyword import KeywordLLMBackend
            fallback = KeywordLLMBackend(self.config)
            return fallback.extract_relations(concepts, text)

    def extract_rules(self, text: str, concepts: list[ConceptNode]) -> list[dict]:
        """Extract rules using Claude."""
        concept_names = [c.name for c in concepts]
        name_to_id = {c.name.lower(): c.id for c in concepts}

        prompt = self.RULE_EXTRACTION_PROMPT.format(
            concepts=", ".join(concept_names),
            text=text[:8000],
        )

        try:
            rules = self._call_claude(prompt, expect_json=True)

            result = []
            for r in rules:
                if isinstance(r, dict) and "text" in r:
                    # Map concept names to IDs
                    concept_ids = []
                    for cname in r.get("concepts", []):
                        cid = name_to_id.get(cname.lower())
                        if cid:
                            concept_ids.append(cid)

                    result.append({
                        "text": r["text"],
                        "concept_ids": concept_ids,
                        "category": r.get("category", "general"),
                        "confidence": float(r.get("confidence", 0.7)),
                    })

            return result[:self.config.max_rules]

        except Exception as e:
            from .keyword import KeywordLLMBackend
            fallback = KeywordLLMBackend(self.config)
            return fallback.extract_rules(text, concepts)

    def suggest_domain_name(self, source_name: str, sample_text: str) -> str:
        """Suggest domain name using Claude."""
        prompt = self.DOMAIN_SUGGESTION_PROMPT.format(
            source_name=source_name,
            sample_text=sample_text[:2000],
        )

        try:
            name = self._call_claude(prompt, expect_json=False)
            # Clean up response
            name = re.sub(r'[^\w_]', '', name.lower())
            return name[:30] if name else "unknown_domain"
        except Exception:
            from .keyword import KeywordLLMBackend
            fallback = KeywordLLMBackend(self.config)
            return fallback.suggest_domain_name(source_name, sample_text)
