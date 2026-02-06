"""Ollama extraction backend for local LLM inference."""

from __future__ import annotations

import json
import os
import re
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from qortex.core.models import ConceptNode


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


class OllamaExtractionBackend:
    """Extraction backend using local Ollama server.

    Implements LLMBackend protocol for concept/relation/rule extraction.
    Requires Ollama running locally (or at OLLAMA_HOST).
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
    ):
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.host = self.host.rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")

    def is_available(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            req = Request(f"{self.host}/api/tags", method="GET")
            with urlopen(req, timeout=2) as resp:
                return resp.status == 200
        except (URLError, TimeoutError, OSError):
            return False

    def _call(self, prompt: str) -> str:
        """Make API call to Ollama and return response."""
        payload = json.dumps(
            {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 4096,
                },
            }
        ).encode("utf-8")

        req = Request(
            f"{self.host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data.get("response", "")
        except (URLError, TimeoutError) as e:
            raise ConnectionError(f"Ollama request failed: {e}") from e

    def _parse_json(self, text: str) -> list | dict:
        """Extract JSON from response, handling markdown code blocks."""
        # Try to find JSON in code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1)

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
        """Extract concepts from text."""
        domain_context = f" in the domain of {domain_hint}" if domain_hint else ""

        prompt = f"""You are a knowledge extraction system. Extract key concepts{domain_context}.

A concept is a distinct idea, pattern, principle, technique, or entity.
Focus on concepts useful for a knowledge graph.

Return ONLY a JSON array of objects with:
- name: Short concept name (2-5 words, title case)
- description: One sentence explaining the concept
- confidence: Float 0-1

Extract 5-15 concepts. No explanation, just JSON.

TEXT:
{text[:6000]}

JSON:"""

        result = self._call(prompt)
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
        """Extract relations between concepts."""
        if not concepts:
            return []

        # Use all concepts (Ollama is local, no API cost)
        concept_list = "\n".join(f"- {c.id}: {c.name}" for c in concepts)

        prompt = f"""You are a knowledge extraction system. Identify relationships between concepts.

CONCEPTS:
{concept_list}

RELATION TYPES (use exactly these):
REQUIRES, CONTRADICTS, REFINES, IMPLEMENTS, PART_OF, USES, SIMILAR_TO, ALTERNATIVE_TO, SUPPORTS, CHALLENGES

Return ONLY a JSON array of objects with:
- source_id: ID from the list above
- target_id: ID from the list above
- relation_type: One of the relation types
- confidence: Float 0-1
- source_text: Quote from text supporting this relation

Aim for 3-5 relations per major concept.

TEXT:
{text}

JSON:"""

        result = self._call(prompt)
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
        """Extract explicit rules from text."""
        concept_list = "\n".join(f"- {c.id}: {c.name}" for c in concepts[:30])

        prompt = f"""You are a knowledge extraction system. Extract actionable rules from text.

A rule is: what to do/not do, how things work, best practices, warnings.

CONCEPTS:
{concept_list}

Return ONLY a JSON array of objects with:
- text: The rule as a clear statement (1-2 sentences)
- concept_ids: Array of concept IDs this rule relates to
- category: "best_practice", "warning", "principle", or "guideline"
- confidence: Float 0-1

TEXT:
{text[:4000]}

JSON:"""

        result = self._call(prompt)
        parsed = self._parse_json(result)

        valid_ids = {c.id for c in concepts}

        if isinstance(parsed, list):
            return [
                {
                    "text": r["text"],
                    "concept_ids": [cid for cid in r.get("concept_ids", []) if cid in valid_ids],
                    "category": r.get("category", "principle"),
                    "confidence": float(r.get("confidence", 0.8)),
                }
                for r in parsed
                if isinstance(r, dict) and r.get("text")
            ]
        return []

    def suggest_domain_name(self, source_name: str, sample_text: str) -> str:
        """Suggest a domain name for the source."""
        prompt = f"""Suggest a short domain name for this content.
Requirements: 1-3 words, lowercase, underscores for spaces.
Example: "software_design" or "error_handling"

Source: {source_name}
Sample: {sample_text[:1000]}

Domain name (just the name, nothing else):"""

        result = self._call(prompt)
        name = result.strip().lower().split()[0] if result.strip() else "unknown"
        name = re.sub(r"[^a-z0-9_]", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name or "unknown"
