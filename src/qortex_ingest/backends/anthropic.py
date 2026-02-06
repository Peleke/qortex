"""Anthropic extraction backend using Claude API."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qortex.core.models import ConceptNode

# Lazy import to avoid hard dependency
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


RELATION_TYPES = [
    "REQUIRES",
    "CONTRADICTS",
    "REFINES",
    "IMPLEMENTS",
    "PART_OF",
    "USES",
    "SIMILAR_TO",
    "ALTERNATIVE_TO",
    "SUPPORTS",
    "CHALLENGES",
]


class AnthropicExtractionBackend:
    """Extraction backend using Anthropic Claude API.

    Implements LLMBackend protocol for concept/relation/rule extraction.
    """

    def __init__(
        self,
        api_key: str,
        model: str | None = None,
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or "claude-sonnet-4-20250514"

    def _call(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """Make API call and return text response."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text

    def _parse_json(self, text: str) -> list | dict:
        """Extract JSON from response, handling markdown code blocks."""
        # Try to find JSON in code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1)

        # Try direct parse
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find array or object
            for start, end in [("[", "]"), ("{", "}")]:
                idx_start = text.find(start)
                idx_end = text.rfind(end)
                if idx_start != -1 and idx_end > idx_start:
                    try:
                        return json.loads(text[idx_start : idx_end + 1])
                    except json.JSONDecodeError:
                        continue
            return []

    def extract_concepts(
        self,
        text: str,
        domain_hint: str | None = None,
    ) -> list[dict]:
        """Extract concepts from text.

        Returns list of dicts with keys: name, description, confidence
        """
        domain_context = f" in the domain of {domain_hint}" if domain_hint else ""

        system = f"""You are a knowledge extraction system. Extract key concepts{domain_context}.

A concept is a distinct idea, pattern, principle, technique, or entity mentioned in the text.
Focus on concepts that would be useful in a knowledge graph for understanding relationships.

Return JSON array of objects with:
- name: Short concept name (2-5 words, title case)
- description: One sentence explaining the concept
- confidence: Float 0-1, how confident you are this is a meaningful concept

Extract 5-20 concepts depending on text length. Prefer quality over quantity."""

        user = f"""Extract concepts from this text:

{text[:8000]}

Return only valid JSON array."""

        result = self._call(system, user)
        parsed = self._parse_json(result)

        if isinstance(parsed, list):
            return [
                {
                    "name": c.get("name", "Unknown"),
                    "description": c.get("description", ""),
                    "confidence": float(c.get("confidence", 0.8)),
                }
                for c in parsed
                if isinstance(c, dict) and c.get("name")
            ]
        return []

    def extract_relations(
        self,
        concepts: list[ConceptNode],
        text: str,
        chunk_location: str | None = None,
    ) -> list[dict]:
        """Extract relations between concepts.

        Returns list of dicts with keys: source_id, target_id, relation_type,
        confidence, source_text, source_location
        """
        if not concepts:
            return []

        # No concept limit - use all concepts (was: concepts[:50])
        concept_list = "\n".join(
            f"- {c.id}: {c.name} - {c.description}" for c in concepts
        )

        relation_list = "\n".join(f"- {r}" for r in RELATION_TYPES)

        system = f"""You are a knowledge extraction system. Identify relationships between concepts.

Available relation types:
{relation_list}

Relation meanings:
- REQUIRES: A needs B to function or exist (dependency)
- CONTRADICTS: A and B are mutually exclusive or incompatible
- REFINES: A is a more specific or detailed version of B
- IMPLEMENTS: A is a concrete realization of abstract B
- PART_OF: A is a component or element of B
- USES: A utilizes or depends on B (weaker than REQUIRES)
- SIMILAR_TO: A and B share significant characteristics
- ALTERNATIVE_TO: A can substitute for B in some contexts
- SUPPORTS: A provides evidence or justification for B
- CHALLENGES: A raises problems or counterarguments for B

Return JSON array of objects with:
- source_id: ID of source concept (exact match from list)
- target_id: ID of target concept (exact match from list)
- relation_type: One of the relation types above (uppercase)
- confidence: Float 0-1 based on how explicitly the text states this
- source_text: 1-2 sentence quote from the text that supports this relation

Aim for 3-5 relations per major concept mentioned in the text.
Only return relationships clearly supported by the text - include the source_text that justifies each."""

        # No text truncation - use full text (was: text[:6000])
        user = f"""Given these concepts:
{concept_list}

And this text:
{text}

Extract relationships between the concepts. For each relationship, include the source_text quote that supports it.
Return only valid JSON array."""

        result = self._call(system, user, max_tokens=8192)
        parsed = self._parse_json(result)

        valid_ids = {c.id for c in concepts}

        if isinstance(parsed, list):
            return [
                {
                    "source_id": r["source_id"],
                    "target_id": r["target_id"],
                    "relation_type": r["relation_type"],
                    "confidence": float(r.get("confidence", 0.8)),
                    "source_text": r.get("source_text", ""),
                    "source_location": chunk_location,
                }
                for r in parsed
                if isinstance(r, dict)
                and r.get("source_id") in valid_ids
                and r.get("target_id") in valid_ids
                and r.get("relation_type") in RELATION_TYPES
            ]
        return []

    def extract_rules(
        self,
        text: str,
        concepts: list[ConceptNode],
    ) -> list[dict]:
        """Extract explicit rules from text.

        Returns list of dicts with keys: text, concept_ids, category, confidence
        """
        concept_list = "\n".join(f"- {c.id}: {c.name}" for c in concepts[:50])

        system = """You are a knowledge extraction system. Extract actionable rules, principles, or guidelines from text.

A rule is an explicit statement about:
- What to do or not do
- How things work or relate
- Best practices or warnings
- Conditions and their consequences

Return JSON array of objects with:
- text: The rule as a clear, actionable statement (1-2 sentences)
- concept_ids: Array of concept IDs this rule relates to
- category: Category like "best_practice", "warning", "principle", "guideline"
- confidence: Float 0-1

Extract rules that are explicitly stated or strongly implied. Do not invent rules."""

        user = f"""Given these concepts:
{concept_list}

And this text:
{text[:6000]}

Extract explicit rules and principles. Return only valid JSON array."""

        result = self._call(system, user)
        parsed = self._parse_json(result)

        valid_ids = {c.id for c in concepts}

        if isinstance(parsed, list):
            return [
                {
                    "text": r["text"],
                    "concept_ids": [
                        cid for cid in r.get("concept_ids", []) if cid in valid_ids
                    ],
                    "category": r.get("category", "principle"),
                    "confidence": float(r.get("confidence", 0.8)),
                }
                for r in parsed
                if isinstance(r, dict) and r.get("text")
            ]
        return []

    def suggest_domain_name(self, source_name: str, sample_text: str) -> str:
        """Suggest a domain name for the source."""
        system = """You are a knowledge organization system. Suggest a short domain name for categorizing content.

The domain name should be:
- 1-3 words, lowercase, underscores for spaces
- Descriptive of the main topic
- Suitable as a database schema name

Return only the domain name, nothing else."""

        user = f"""Source: {source_name}

Sample content:
{sample_text[:2000]}

Suggest a domain name:"""

        result = self._call(system, user, max_tokens=50)
        # Clean up the response
        name = result.strip().lower()
        name = re.sub(r"[^a-z0-9_]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name or "unknown"
