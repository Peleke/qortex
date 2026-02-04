"""Keyword-based extraction backend (no LLM required).

Useful for:
- Testing without API costs
- Offline operation
- Fast extraction with lower quality
"""

from __future__ import annotations

import re
from collections import Counter

from qortex.core.models import ConceptNode, RelationType

from .base import BaseLLMBackend, LLMConfig


class KeywordLLMBackend(BaseLLMBackend):
    """Simple keyword-based extraction for testing.

    Extracts concepts based on:
    - Capitalized phrases
    - Quoted terms
    - Repeated important words

    Relations are inferred from proximity and keywords.
    """

    # Common relation indicators
    RELATION_INDICATORS = {
        RelationType.CONTRADICTS: ["not", "avoid", "instead of", "rather than", "opposite", "contrary"],
        RelationType.REQUIRES: ["requires", "needs", "depends on", "must have", "prerequisite", "necessary"],
        RelationType.REFINES: ["specifically", "more precisely", "in particular", "a type of", "subtype"],
        RelationType.IMPLEMENTS: ["implements", "realizes", "achieves", "accomplishes", "provides"],
        RelationType.SIMILAR_TO: ["similar to", "like", "analogous", "related to", "comparable"],
        RelationType.PART_OF: ["part of", "component of", "belongs to", "within", "contained in"],
        RelationType.USES: ["uses", "utilizes", "employs", "applies", "leverages"],
        RelationType.ALTERNATIVE_TO: ["alternative", "or", "instead", "substitute", "replacement"],
    }

    # Stop words to filter
    STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "each", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
        "because", "until", "while", "this", "that", "these", "those", "what",
        "which", "who", "whom", "whose", "it", "its", "they", "them", "their",
        "also", "about", "your", "you", "we", "our", "my", "me", "him", "her",
    }

    def __init__(self, config: LLMConfig | None = None):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "keyword"

    @property
    def is_available(self) -> bool:
        return True  # Always available - no external dependencies

    def extract_concepts(self, text: str, domain_hint: str | None = None) -> list[dict]:
        """Extract concepts from text using keyword analysis."""
        concepts = []
        seen = set()
        max_concepts = self.config.max_concepts

        # 1. Extract capitalized phrases (potential concepts)
        cap_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        for match in re.finditer(cap_pattern, text):
            name = match.group(1)
            if name.lower() not in self.STOP_WORDS and name not in seen:
                seen.add(name)
                concepts.append({
                    "name": name,
                    "description": self._get_context(text, match.start(), 100),
                    "confidence": 0.8,
                })

        # 2. Extract quoted terms
        quote_pattern = r'["\']([^"\']+)["\']'
        for match in re.finditer(quote_pattern, text):
            term = match.group(1).strip()
            if len(term) > 2 and len(term) < 50 and term not in seen:
                seen.add(term)
                concepts.append({
                    "name": term,
                    "description": self._get_context(text, match.start(), 100),
                    "confidence": 0.9,
                })

        # 3. Extract frequently repeated words (important concepts)
        words = re.findall(r'\b([a-z]{4,})\b', text.lower())
        word_counts = Counter(words)
        for word, count in word_counts.most_common(15):
            if count >= 3 and word not in self.STOP_WORDS and word not in seen:
                seen.add(word)
                concepts.append({
                    "name": word.title(),
                    "description": f"Frequently mentioned ({count} times)",
                    "confidence": min(0.5 + count * 0.1, 0.9),
                })

        # Filter by minimum confidence
        concepts = [c for c in concepts if c["confidence"] >= self.config.min_confidence]

        return concepts[:max_concepts]

    def extract_relations(
        self,
        concepts: list[ConceptNode],
        text: str,
    ) -> list[dict]:
        """Extract relations between concepts based on text proximity and keywords."""
        relations = []
        text_lower = text.lower()

        # For each pair of concepts, check for relation indicators
        concept_list = list(concepts)
        for i, c1 in enumerate(concept_list):
            for c2 in concept_list[i + 1:]:
                # Check proximity in text
                c1_pos = text_lower.find(c1.name.lower())
                c2_pos = text_lower.find(c2.name.lower())

                if c1_pos >= 0 and c2_pos >= 0:
                    # Get text between concepts
                    start, end = min(c1_pos, c2_pos), max(c1_pos, c2_pos)
                    between = text_lower[start:end + len(c2.name)]

                    # Check for relation indicators
                    found_relation = False
                    for rel_type, indicators in self.RELATION_INDICATORS.items():
                        for indicator in indicators:
                            if indicator in between:
                                relations.append({
                                    "source_id": c1.id,
                                    "target_id": c2.id,
                                    "relation_type": rel_type,
                                    "confidence": 0.6,
                                })
                                found_relation = True
                                break
                        if found_relation:
                            break

                    # Default: if close together, assume similarity
                    if not found_relation and abs(c1_pos - c2_pos) < 100:
                        relations.append({
                            "source_id": c1.id,
                            "target_id": c2.id,
                            "relation_type": RelationType.SIMILAR_TO,
                            "confidence": 0.4,
                        })

        return relations

    def extract_rules(self, text: str, concepts: list[ConceptNode]) -> list[dict]:
        """Extract explicit rules from text."""
        rules = []
        concept_ids = {c.name.lower(): c.id for c in concepts}
        max_rules = self.config.max_rules

        # Imperative patterns
        imperative_patterns = [
            r'(?:^|\.\s+)((?:Always|Never|Avoid|Use|Prefer|Consider|Ensure|Make sure)[^.!?]+[.!?])',
            r'(?:^|\.\s+)((?:Do not|Don\'t|Should|Must|Shall)[^.!?]+[.!?])',
            r'(?:^|\n)\s*[-•]\s*([^.\n]+[.!?]?)',  # Bullet points
            r'(?:^|\n)\s*\d+[.)]\s*([^.\n]+[.!?]?)',  # Numbered lists
        ]

        for pattern in imperative_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE):
                rule_text = match.group(1).strip()
                if len(rule_text) > 20 and len(rule_text) < 200:
                    # Find related concepts
                    related_concepts = []
                    rule_lower = rule_text.lower()
                    for name, cid in concept_ids.items():
                        if name in rule_lower:
                            related_concepts.append(cid)

                    rules.append({
                        "text": rule_text,
                        "concept_ids": related_concepts,
                        "category": self._categorize_rule(rule_text),
                        "confidence": 0.7,
                    })

        return rules[:max_rules]

    def suggest_domain_name(self, source_name: str, sample_text: str) -> str:
        """Suggest a domain name based on source and content."""
        # Clean up source name
        name = re.sub(r'[^\w\s]', '', source_name)
        name = name.lower().replace(' ', '_')

        # If too generic, try to extract from content
        if len(name) < 3 or name in ['book', 'chapter', 'text', 'document']:
            words = re.findall(r'\b([a-z]{5,})\b', sample_text.lower())
            if words:
                word_counts = Counter(words)
                for word, _ in word_counts.most_common(5):
                    if word not in self.STOP_WORDS:
                        return f"{word}_domain"

        return name[:30]

    def _get_context(self, text: str, position: int, window: int) -> str:
        """Get context around a position in text."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        context = text[start:end].strip()
        return f"...{context}..." if start > 0 or end < len(text) else context

    def _categorize_rule(self, rule_text: str) -> str:
        """Categorize a rule based on its content."""
        rule_lower = rule_text.lower()

        if any(w in rule_lower for w in ['test', 'assert', 'verify', 'check', 'coverage']):
            return "testing"
        if any(w in rule_lower for w in ['security', 'auth', 'permission', 'encrypt', 'vulnerability']):
            return "security"
        if any(w in rule_lower for w in ['function', 'class', 'method', 'interface', 'module', 'pattern']):
            return "architectural"
        if any(w in rule_lower for w in ['avoid', 'never', 'not', "don't", 'anti']):
            return "antipattern"
        if any(w in rule_lower for w in ['performance', 'optimize', 'cache', 'memory', 'speed']):
            return "performance"

        return "general"
