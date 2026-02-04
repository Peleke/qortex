"""Ollama LLM backend for local/private extraction.

Useful for:
- Offline operation
- Privacy-sensitive content
- NSFW or restricted content
- Cost control
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from qortex.core.models import ConceptNode, RelationType

from .base import BaseLLMBackend, LLMConfig

# Lazy import
_ollama_client = None


def _get_ollama_client(base_url: str | None = None):
    """Lazy-load ollama client."""
    global _ollama_client
    if _ollama_client is None:
        try:
            import ollama
            # Ollama uses environment or default localhost
            _ollama_client = ollama
        except ImportError:
            pass
    return _ollama_client


def _check_ollama_running(base_url: str | None = None) -> bool:
    """Check if Ollama service is running."""
    try:
        import httpx
        url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        response = httpx.get(f"{url}/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


class OllamaLLMBackend(BaseLLMBackend):
    """Local Ollama-powered extraction.

    Uses local models like:
    - llama3
    - mistral
    - codellama
    - phi3
    """

    DEFAULT_MODEL = "llama3"

    # Simplified prompts for smaller models
    CONCEPT_PROMPT = """Extract key concepts from this text as JSON array.
Each concept: {{"name": "...", "description": "...", "confidence": 0.8}}

Text: {text}

JSON:"""

    RELATION_PROMPT = """Given concepts: {concepts}

Find relationships between them from types: contradicts, requires, refines, implements, similar_to, part_of, uses

Return JSON: [{{"source": "name1", "target": "name2", "relation": "type"}}]

Text: {text}

JSON:"""

    RULE_PROMPT = """Extract rules/guidelines from this text as JSON.
Each rule: {{"text": "the rule", "category": "general|testing|security|architectural"}}

Text: {text}

JSON:"""

    def __init__(self, config: LLMConfig | None = None):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "ollama"

    @property
    def is_available(self) -> bool:
        """Check if Ollama is running."""
        return _check_ollama_running(self.config.base_url)

    def _call_ollama(self, prompt: str) -> Any:
        """Make an Ollama API call."""
        client = _get_ollama_client(self.config.base_url)
        if not client:
            raise RuntimeError("Ollama not available")

        model = self.config.model or self.DEFAULT_MODEL

        response = client.generate(
            model=model,
            prompt=prompt,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        )

        text = response["response"]

        # Try to extract JSON
        json_match = re.search(r'(\[[\s\S]*?\]|\{[\s\S]*?\})', text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        return text

    def extract_concepts(self, text: str, domain_hint: str | None = None) -> list[dict]:
        """Extract concepts using Ollama."""
        prompt = self.CONCEPT_PROMPT.format(text=text[:4000])

        try:
            result = self._call_ollama(prompt)
            if isinstance(result, list):
                return [
                    {
                        "name": c.get("name", ""),
                        "description": c.get("description", ""),
                        "confidence": float(c.get("confidence", 0.7)),
                    }
                    for c in result
                    if isinstance(c, dict) and c.get("name")
                ][:self.config.max_concepts]
        except Exception:
            pass

        # Fallback
        from .keyword import KeywordLLMBackend
        return KeywordLLMBackend(self.config).extract_concepts(text, domain_hint)

    def extract_relations(
        self,
        concepts: list[ConceptNode],
        text: str,
    ) -> list[dict]:
        """Extract relations using Ollama."""
        concept_names = ", ".join(c.name for c in concepts[:10])
        name_to_id = {c.name.lower(): c.id for c in concepts}

        prompt = self.RELATION_PROMPT.format(
            concepts=concept_names,
            text=text[:4000],
        )

        try:
            result = self._call_ollama(prompt)
            if isinstance(result, list):
                relations = []
                for r in result:
                    if isinstance(r, dict):
                        source_id = name_to_id.get(r.get("source", "").lower())
                        target_id = name_to_id.get(r.get("target", "").lower())
                        if source_id and target_id:
                            try:
                                rel_type = RelationType(r.get("relation", "similar_to").lower())
                            except ValueError:
                                rel_type = RelationType.SIMILAR_TO
                            relations.append({
                                "source_id": source_id,
                                "target_id": target_id,
                                "relation_type": rel_type,
                                "confidence": 0.6,
                            })
                return relations
        except Exception:
            pass

        from .keyword import KeywordLLMBackend
        return KeywordLLMBackend(self.config).extract_relations(concepts, text)

    def extract_rules(self, text: str, concepts: list[ConceptNode]) -> list[dict]:
        """Extract rules using Ollama."""
        prompt = self.RULE_PROMPT.format(text=text[:4000])
        name_to_id = {c.name.lower(): c.id for c in concepts}

        try:
            result = self._call_ollama(prompt)
            if isinstance(result, list):
                rules = []
                for r in result:
                    if isinstance(r, dict) and r.get("text"):
                        # Find related concepts
                        concept_ids = []
                        rule_lower = r["text"].lower()
                        for name, cid in name_to_id.items():
                            if name in rule_lower:
                                concept_ids.append(cid)

                        rules.append({
                            "text": r["text"],
                            "concept_ids": concept_ids,
                            "category": r.get("category", "general"),
                            "confidence": 0.6,
                        })
                return rules[:self.config.max_rules]
        except Exception:
            pass

        from .keyword import KeywordLLMBackend
        return KeywordLLMBackend(self.config).extract_rules(text, concepts)

    def suggest_domain_name(self, source_name: str, sample_text: str) -> str:
        """Use keyword backend for domain suggestion (faster)."""
        from .keyword import KeywordLLMBackend
        return KeywordLLMBackend(self.config).suggest_domain_name(source_name, sample_text)
