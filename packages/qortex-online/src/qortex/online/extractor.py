"""Pluggable concept extraction for online session indexing.

Defines an ExtractionStrategy protocol and three implementations:
  - SpaCyExtractor (default): NER + noun chunks, fast, local, no API key
  - LLMExtractor (opt-in): wraps existing qortex-ingest LLMBackend
  - NullExtractor: explicit no-op (QORTEX_EXTRACTION=none)

Every operation is span-traced for Jaeger/Grafana visibility.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractedConcept:
    """A named concept extracted from text."""

    name: str  # e.g. "JWT Tokens", "Auth Module"
    description: str  # One-sentence context
    confidence: float = 1.0


@dataclass(frozen=True)
class ExtractedRelation:
    """A typed relationship between two extracted concepts."""

    source_name: str
    target_name: str
    relation_type: str  # Maps to RelationType enum values
    confidence: float = 0.8


@dataclass(frozen=True)
class ExtractionResult:
    """Output of an extraction strategy."""

    concepts: list[ExtractedConcept] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.concepts


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ExtractionStrategy(Protocol):
    """Protocol for concept extraction strategies.

    Any callable with this signature can be used as an extractor.
    """

    def __call__(self, text: str, domain: str = "") -> ExtractionResult: ...


# ---------------------------------------------------------------------------
# SpaCy Extractor (default)
# ---------------------------------------------------------------------------

# Entity labels we care about for concept extraction
_CONCEPT_LABELS = frozenset({
    "PERSON", "ORG", "PRODUCT", "GPE", "WORK_OF_ART", "EVENT",
    "FAC", "LAW", "LANGUAGE", "NORP",
})

# Dependency-parse relation inference patterns
_VERB_USES = frozenset({"use", "utilize", "call", "invoke"})
_VERB_REQUIRES = frozenset({
    "require", "need", "depend", "import",
})


def _title_case(text: str) -> str:
    """Normalize concept name to title case, stripping excess whitespace."""
    return re.sub(r"\s+", " ", text.strip()).title()


def _deduplicate_spans(
    entities: list[tuple[int, int, str, str]],
    noun_chunks: list[tuple[int, int, str]],
) -> list[ExtractedConcept]:
    """Merge NER entities and noun chunks, preferring entities on overlap."""
    occupied: set[int] = set()
    concepts: list[ExtractedConcept] = []

    for start, end, text, label in entities:
        name = _title_case(text)
        if not name or len(name) < 2:
            continue
        concepts.append(ExtractedConcept(
            name=name,
            description=f"{label.title()} entity: {text.strip()}",
            confidence=0.9,
        ))
        occupied.update(range(start, end))

    seen_names: set[str] = {c.name.lower() for c in concepts}
    for start, end, text in noun_chunks:
        name = _title_case(text)
        if not name or len(name) < 2 or name.lower() in seen_names:
            continue
        if any(i in occupied for i in range(start, end)):
            continue
        concepts.append(ExtractedConcept(
            name=name,
            description=f"Noun phrase: {text.strip()}",
            confidence=0.7,
        ))
        seen_names.add(name.lower())

    return concepts


class SpaCyExtractor:
    """Default extraction strategy: spaCy NER + noun chunks.

    Eagerly downloads en_core_web_sm on first call.
    Falls back gracefully if spaCy is not installed (returns empty).
    """

    def __init__(self) -> None:
        self._nlp: Any = None
        self._available: bool | None = None

    def _ensure_loaded(self) -> bool:
        """Load spaCy model, downloading eagerly if not present."""
        if self._available is not None:
            return self._available

        try:
            import spacy
        except ImportError:
            logger.warning(
                "spaCy not installed — extraction disabled. "
                "Install with: uv pip install 'qortex-online[nlp]'"
            )
            self._available = False
            return False

        try:
            self._nlp = spacy.load("en_core_web_sm")
            self._available = True
            logger.info("spaCy en_core_web_sm loaded successfully")
            return True
        except OSError:
            # Model not found — download eagerly
            logger.info("Downloading spaCy en_core_web_sm model...")
            t0 = time.monotonic()
            try:
                spacy.cli.download("en_core_web_sm")  # type: ignore[attr-defined]
                self._nlp = spacy.load("en_core_web_sm")
                elapsed = (time.monotonic() - t0) * 1000
                logger.info("spaCy model downloaded and loaded in %.0fms", elapsed)
                self._available = True
                return True
            except Exception:
                logger.exception("Failed to download spaCy model — extraction disabled")
                self._available = False
                return False

    def __call__(self, text: str, domain: str = "") -> ExtractionResult:
        """Extract concepts and relations from text using spaCy."""
        if not text or not text.strip():
            return ExtractionResult()

        if not self._ensure_loaded():
            return ExtractionResult()

        try:
            return self._extract_traced(text, domain)
        except Exception:
            logger.exception("spaCy extraction failed — returning empty result")
            return ExtractionResult()

    def _extract_traced(self, text: str, domain: str) -> ExtractionResult:
        """Top-level traced extraction."""
        try:
            from qortex.observe.tracing import traced

            @traced("extraction.spacy")
            def _run() -> ExtractionResult:
                return self._extract_inner(text, domain)

            return _run()
        except ImportError:
            return self._extract_inner(text, domain)

    def _extract_inner(self, text: str, domain: str) -> ExtractionResult:
        """Inner extraction: NER + noun chunks + dependency relations."""
        try:
            from qortex.observe.tracing import traced
            _traced = traced
        except ImportError:
            # No tracing available — run without spans
            def _traced(name: str, **kw):  # type: ignore[assignment]
                def dec(fn):  # type: ignore[no-untyped-def]
                    return fn
                return dec

        # --- Span: NLP processing ---
        @_traced("extraction.spacy.nlp_process")
        def _nlp_process():  # type: ignore[no-untyped-def]
            return self._nlp(text)

        doc = _nlp_process()

        # --- Span: Entity extraction ---
        @_traced("extraction.spacy.extract_entities")
        def _extract_entities():  # type: ignore[no-untyped-def]
            entities = []
            for ent in doc.ents:
                if ent.label_ in _CONCEPT_LABELS:
                    entities.append((ent.start_char, ent.end_char, ent.text, ent.label_))
            return entities

        entities = _extract_entities()

        # --- Span: Noun chunk extraction ---
        @_traced("extraction.spacy.extract_noun_chunks")
        def _extract_noun_chunks():  # type: ignore[no-untyped-def]
            chunks = []
            for chunk in doc.noun_chunks:
                if len(chunk) == 1 and chunk[0].pos_ in ("PRON", "DET"):
                    continue
                chunks.append((chunk.start_char, chunk.end_char, chunk.text))
            return chunks

        noun_chunks = _extract_noun_chunks()

        # --- Span: Deduplication ---
        @_traced("extraction.spacy.deduplicate")
        def _dedup():  # type: ignore[no-untyped-def]
            return _deduplicate_spans(entities, noun_chunks)

        concepts = _dedup()

        if not concepts:
            return ExtractionResult()

        # --- Span: Relation inference ---
        @_traced("extraction.spacy.infer_relations")
        def _infer():  # type: ignore[no-untyped-def]
            return self._infer_relations_from_doc(doc, concepts)

        relations = _infer()

        return ExtractionResult(concepts=concepts, relations=relations)

    def _infer_relations_from_doc(
        self,
        doc: Any,
        concepts: list[ExtractedConcept],
    ) -> list[ExtractedRelation]:
        """Infer typed relations from spaCy dependency parse."""
        concept_names = {c.name.lower() for c in concepts}
        relations: list[ExtractedRelation] = []
        seen: set[tuple[str, str, str]] = set()

        for sent in doc.sents:
            subj: str | None = None
            verb: Any = None

            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    for nc in doc.noun_chunks:
                        if token.i >= nc.start and token.i < nc.end:
                            chunk_text = _title_case(nc.text)
                            if chunk_text.lower() in concept_names:
                                subj = chunk_text
                                verb = token.head
                            break

                if token.dep_ in ("dobj", "pobj", "attr"):
                    for nc in doc.noun_chunks:
                        if token.i >= nc.start and token.i < nc.end:
                            obj_text = _title_case(nc.text)
                            if obj_text.lower() in concept_names and subj and verb:
                                rel_type = self._verb_to_relation(verb.lemma_)
                                key = (subj, obj_text, rel_type)
                                if key not in seen:
                                    relations.append(ExtractedRelation(
                                        source_name=subj,
                                        target_name=obj_text,
                                        relation_type=rel_type,
                                        confidence=0.6,
                                    ))
                                    seen.add(key)
                            break

            # Coordination patterns ("X and Y" → SIMILAR_TO)
            for token in sent:
                if token.dep_ == "conj" and token.head:
                    head_text = _title_case(token.head.text)
                    conj_text = _title_case(token.text)
                    if (
                        head_text.lower() in concept_names
                        and conj_text.lower() in concept_names
                    ):
                        key = (head_text, conj_text, "SIMILAR_TO")
                        if key not in seen:
                            relations.append(ExtractedRelation(
                                source_name=head_text,
                                target_name=conj_text,
                                relation_type="SIMILAR_TO",
                                confidence=0.5,
                            ))
                            seen.add(key)

        return relations

    @staticmethod
    def _verb_to_relation(lemma: str) -> str:
        """Map a verb lemma to a relation type."""
        if lemma in _VERB_USES:
            return "USES"
        if lemma in _VERB_REQUIRES:
            return "REQUIRES"
        if lemma in ("contain", "include", "have", "hold"):
            return "CONTAINS"
        if lemma in ("implement", "extend", "inherit"):
            return "IMPLEMENTS"
        if lemma in ("refine", "specialize", "customize"):
            return "REFINES"
        return "RELATED_TO"


# ---------------------------------------------------------------------------
# LLM Extractor (opt-in)
# ---------------------------------------------------------------------------


class LLMExtractor:
    """Extraction strategy wrapping existing qortex-ingest LLMBackend.

    Opt-in via QORTEX_EXTRACTION=llm. Uses the same Anthropic/Ollama
    backends that power batch ingestion.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend

    def __call__(self, text: str, domain: str = "") -> ExtractionResult:
        """Extract concepts and relations via LLM backend."""
        if not text or not text.strip():
            return ExtractionResult()

        try:
            return self._extract_traced(text, domain)
        except Exception:
            logger.exception("LLM extraction failed — returning empty result")
            return ExtractionResult()

    def _extract_traced(self, text: str, domain: str) -> ExtractionResult:
        """Top-level traced extraction."""
        try:
            from qortex.observe.tracing import traced

            @traced("extraction.llm", external=True)
            def _run() -> ExtractionResult:
                return self._extract_inner(text, domain)

            return _run()
        except ImportError:
            return self._extract_inner(text, domain)

    def _extract_inner(self, text: str, domain: str) -> ExtractionResult:
        """Call LLMBackend methods with per-step tracing."""
        try:
            from qortex.observe.tracing import traced
            _traced = traced
        except ImportError:
            def _traced(name: str, **kw):  # type: ignore[assignment]
                def dec(fn):  # type: ignore[no-untyped-def]
                    return fn
                return dec

        # --- Span: Concept extraction ---
        @_traced("extraction.llm.extract_concepts", external=True)
        def _extract_concepts():  # type: ignore[no-untyped-def]
            return self._backend.extract_concepts(text, domain_hint=domain or None)

        raw_concepts = _extract_concepts()

        concepts = [
            ExtractedConcept(
                name=c.get("name", "Unknown"),
                description=c.get("description", ""),
                confidence=float(c.get("confidence", 0.8)),
            )
            for c in raw_concepts
            if c.get("name")
        ]

        if not concepts:
            return ExtractionResult()

        # --- Span: Relation extraction ---
        @_traced("extraction.llm.extract_relations", external=True)
        def _extract_relations():  # type: ignore[no-untyped-def]
            from qortex.core.models import ConceptNode

            temp_nodes = [
                ConceptNode(
                    id=f"temp:{i}",
                    name=c.name,
                    description=c.description,
                    domain=domain,
                    source_id="extraction",
                )
                for i, c in enumerate(concepts)
            ]
            return self._backend.extract_relations(temp_nodes, text)

        raw_relations = _extract_relations()

        relations = [
            ExtractedRelation(
                source_name=r.get("source", ""),
                target_name=r.get("target", ""),
                relation_type=r.get("relation_type", "RELATED_TO"),
                confidence=float(r.get("confidence", 0.7)),
            )
            for r in raw_relations
            if r.get("source") and r.get("target")
        ]

        return ExtractionResult(concepts=concepts, relations=relations)


# ---------------------------------------------------------------------------
# Null Extractor (explicit no-op for QORTEX_EXTRACTION=none)
# ---------------------------------------------------------------------------


class NullExtractor:
    """No-op extractor. Pipeline uses raw text[:80] as before."""

    def __call__(self, text: str, domain: str = "") -> ExtractionResult:
        return ExtractionResult()
