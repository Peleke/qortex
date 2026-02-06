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
        max_concepts_per_call: int = 100,
    ):
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "anthropic package not installed. "
                "Install with: pip install anthropic"
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or "claude-sonnet-4-20250514"
        self.max_concepts_per_call = max_concepts_per_call

    def _call(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """Make API call with rate limit retry."""
        import time

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return response.content[0].text
            except anthropic.RateLimitError as e:
                if attempt < max_retries - 1:
                    wait_time = 60 * (attempt + 1)  # 60s, 120s, 180s
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

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

        # Limit concepts to avoid rate limits (100 concepts ~ 10K tokens)
        limited_concepts = concepts[:self.max_concepts_per_call]
        if len(concepts) > self.max_concepts_per_call:
            print(f"  (limiting to {self.max_concepts_per_call}/{len(concepts)} concepts)")

        concept_list = "\n".join(
            f"- {c.id}: {c.name} - {c.description}" for c in limited_concepts
        )

        system = """You are a precise knowledge graph extraction system.

Your task is to identify TEXT-SUPPORTED relationships between the provided concepts, based strictly on the given text.
Do NOT rely on outside knowledge. If a relationship is not clearly stated or directly supported by the text, do not extract it.

RELATION TYPES (with disambiguation)

REQUIRES
- Use when A cannot function, exist, or be correctly applied without B.
- Strong dependency; removal of B breaks A.
- NOT the same as USES (which may be optional or contextual).

USES
- Use when A leverages, depends on, or commonly employs B, but could exist without it.
- Prefer USES when the dependency is practical rather than structural.

REFINES
- Use when A is a more specific, constrained, or specialized form of B.
- A narrows the scope or adds detail to B.
- NOT an implementation.

IMPLEMENTS
- Use when A is a concrete realization, mechanism, or technique that puts B into practice.
- B is abstract or conceptual; A makes it operational.

PART_OF
- Use when A is a component, sub-process, or constituent of B.
- Structural or compositional relationship.

SIMILAR_TO
- Use when A and B address the same problem or role in a comparable way.
- They may coexist; neither replaces the other.

ALTERNATIVE_TO
- Use when A and B serve similar purposes but are positioned as substitutes or competing choices.
- Often signaled by "instead of," "can be replaced by," or explicit comparison.

SUPPORTS
- Use when A provides evidence, justification, or rationale for B.
- Often argumentative or explanatory.

CHALLENGES
- Use when A questions, weakens, or argues against B.
- Includes critiques, limitations, or counterexamples.

CONTRADICTS
- Use only when A and B are explicitly stated to be incompatible or mutually exclusive.

DENSITY & COVERAGE GUIDANCE

- Aim for 3-5 relationships per major concept when supported by the text.
- Every significant concept should have at least one incoming or outgoing edge, if the text allows.
- Prefer multiple precise edges over a few overly conservative ones.
- Do not invent relations to satisfy density; precision remains mandatory.

OUTPUT FORMAT

Return a valid JSON array of objects with:
- source_id: ID of the source concept (from the provided list)
- target_id: ID of the target concept (from the provided list)
- relation_type: One of the relation types above (uppercase)
- confidence: Float 0-1 indicating extraction confidence
- source_text: 1-2 sentence quote from the text that directly supports the relationship

FEW-SHOT EXAMPLES

Example 1:
{"source_id": "design:Encapsulation", "target_id": "design:Information Hiding", "relation_type": "IMPLEMENTS", "confidence": 0.92, "source_text": "Encapsulation implements information hiding by bundling data with the methods that operate on that data."}

Example 2:
{"source_id": "design:Dependency Injection", "target_id": "design:Loose Coupling", "relation_type": "SUPPORTS", "confidence": 0.88, "source_text": "By supplying dependencies from the outside, dependency injection promotes loose coupling between components."}

Example 3:
{"source_id": "design:Inheritance", "target_id": "design:Composition", "relation_type": "ALTERNATIVE_TO", "confidence": 0.85, "source_text": "Many designers recommend composition over inheritance as a more flexible alternative."}

Example 4:
{"source_id": "design:Interface", "target_id": "design:Abstraction", "relation_type": "REFINES", "confidence": 0.83, "source_text": "Interfaces are a specific form of abstraction that define behavior without implementation."}

Example 5:
{"source_id": "design:Unit Testing", "target_id": "design:Testability", "relation_type": "SUPPORTS", "confidence": 0.9, "source_text": "Writing unit tests increases testability by forcing components to be isolated and observable."}

FINAL INSTRUCTIONS

- Extract only relationships clearly supported by the text.
- Prefer explicit statements over vague implications.
- Return ONLY the JSON array. No commentary or explanation."""

        # No text truncation - use full chunk text
        user = f"""Given the following concepts (with IDs, names, and descriptions):

{concept_list}

And the following text:

{text}

Extract all clearly text-supported relationships between the concepts.
Follow the relation definitions, density guidance, and output format exactly.
Return ONLY a valid JSON array."""

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
