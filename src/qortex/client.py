"""QortexClient: the public interface for qortex memory.

All consumer adapters (crewai, langchain, agno, Mastra, MCP) target this
protocol. Third parties build their own adapters against it.

Architecture:
    QortexClient (Protocol)
    ├── LocalQortexClient  — direct in-process, no MCP subprocess
    └── McpQortexClient    — talks to qortex MCP server (future)

Result types have conversion helpers for framework interop:
    QueryItem.to_langchain_document()
    QueryItem.to_crewai_result()
    QueryItem.to_agno_document()
    QueryItem.to_mastra_result()
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class QueryItem:
    """A single result from a qortex query.

    Canonical shape: {id, content, score, domain, node_id, metadata}
    All framework adapters map FROM this shape.
    """

    id: str
    content: str
    score: float
    domain: str
    node_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Framework conversion helpers --

    def to_langchain_document(self) -> Any:
        """Convert to langchain Document(page_content, metadata, id).

        Returns a dict matching langchain's Document shape. If langchain
        is installed, returns an actual Document instance.
        """
        meta = {"score": self.score, "domain": self.domain, "node_id": self.node_id}
        meta.update(self.metadata)
        try:
            from langchain_core.documents import Document

            return Document(page_content=self.content, metadata=meta, id=self.id)
        except ImportError:
            return {"page_content": self.content, "metadata": meta, "id": self.id}

    def to_crewai_result(self) -> dict[str, Any]:
        """Convert to crewai SearchResult shape: {id, content, metadata, score}."""
        meta = {"domain": self.domain, "node_id": self.node_id}
        meta.update(self.metadata)
        return {
            "id": self.id,
            "content": self.content,
            "metadata": meta,
            "score": self.score,
        }

    def to_agno_document(self) -> Any:
        """Convert to agno Document shape.

        Returns a dict matching agno's Document. If agno is installed,
        returns an actual Document instance.
        """
        meta = {"domain": self.domain, "node_id": self.node_id}
        meta.update(self.metadata)
        try:
            from agno.knowledge.document import Document

            return Document(
                content=self.content,
                id=self.id,
                name=self.node_id,
                meta_data=meta,
                reranking_score=self.score,
            )
        except ImportError:
            return {
                "content": self.content,
                "id": self.id,
                "name": self.node_id,
                "meta_data": meta,
                "reranking_score": self.score,
            }

    def to_mastra_result(self) -> dict[str, Any]:
        """Convert to Mastra QueryResult shape: {id, score, metadata, document}."""
        meta = {"domain": self.domain, "node_id": self.node_id}
        meta.update(self.metadata)
        return {
            "id": self.id,
            "score": self.score,
            "metadata": meta,
            "document": self.content,
        }

    def to_autogen_memory_content(self) -> dict[str, Any]:
        """Convert to AutoGen MemoryContent shape.

        Returns a dict with {content, mime_type, metadata}. If autogen-core
        is installed, returns an actual MemoryContent instance.
        """
        meta = {
            "score": self.score,
            "domain": self.domain,
            "node_id": self.node_id,
            "id": self.id,
        }
        meta.update(self.metadata)
        try:
            from autogen_core.memory import MemoryContent, MemoryMimeType

            return MemoryContent(
                content=self.content,
                mime_type=MemoryMimeType.TEXT,
                metadata=meta,
            )
        except ImportError:
            return {
                "content": self.content,
                "mime_type": "text/plain",
                "metadata": meta,
            }


# -- Graph exploration result types --


@dataclass
class NodeItem:
    """A node in the knowledge graph."""

    id: str
    name: str
    description: str
    domain: str
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeItem:
    """A typed edge between two nodes."""

    source_id: str
    target_id: str
    relation_type: str
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleItem:
    """A rule surfaced from the knowledge graph."""

    id: str
    text: str
    domain: str
    category: str | None = None
    confidence: float = 1.0
    relevance: float = 0.0
    derivation: str = "explicit"
    source_concepts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExploreResult:
    """Result of exploring a node's neighborhood."""

    node: NodeItem
    edges: list[EdgeItem] = field(default_factory=list)
    rules: list[RuleItem] = field(default_factory=list)
    neighbors: list[NodeItem] = field(default_factory=list)


@dataclass
class RulesResult:
    """Result of a rules projection query."""

    rules: list[RuleItem]
    domain_count: int = 0
    projection: str = "rules"


@dataclass
class QueryResult:
    """Result of a qortex query."""

    items: list[QueryItem]
    query_id: str
    rules: list[RuleItem] = field(default_factory=list)


@dataclass
class FeedbackResult:
    """Result of submitting feedback."""

    status: str
    query_id: str
    outcome_count: int
    source: str


@dataclass
class IngestResult:
    """Result of ingesting a file."""

    domain: str
    source: str
    concepts: int
    edges: int
    rules: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class DomainInfo:
    """Information about a domain."""

    name: str
    description: str | None = None
    concept_count: int = 0
    edge_count: int = 0
    rule_count: int = 0


@dataclass
class StatusResult:
    """Server/client status."""

    status: str
    backend: str
    vector_index: str | None = None
    vector_search: bool = False
    graph_algorithms: bool = False
    domain_count: int = 0
    embedding_model: str | None = None


# ---------------------------------------------------------------------------
# Helpers: convert core models → client result types
# ---------------------------------------------------------------------------


def _node_to_item(node: Any) -> NodeItem:
    """Convert a ConceptNode to NodeItem."""
    return NodeItem(
        id=node.id,
        name=node.name,
        description=node.description,
        domain=node.domain,
        confidence=node.confidence,
        properties=node.properties,
    )


def _edge_to_item(edge: Any) -> EdgeItem:
    """Convert a ConceptEdge to EdgeItem."""
    return EdgeItem(
        source_id=edge.source_id,
        target_id=edge.target_id,
        relation_type=edge.relation_type.value
        if hasattr(edge.relation_type, "value")
        else str(edge.relation_type),
        confidence=edge.confidence,
        properties=edge.properties,
    )


def _rule_to_item(rule: Any) -> RuleItem:
    """Convert a core Rule to RuleItem."""
    return RuleItem(
        id=rule.id,
        text=rule.text,
        domain=rule.domain,
        category=rule.category,
        confidence=rule.confidence,
        relevance=rule.relevance,
        derivation=rule.derivation,
        source_concepts=rule.source_concepts,
        metadata=rule.metadata,
    )


# ---------------------------------------------------------------------------
# QortexClient protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class QortexClient(Protocol):
    """The public interface for qortex memory.

    Python consumers implement this directly or use LocalQortexClient.
    Non-Python consumers use the MCP server (same semantics).
    Third parties build framework adapters against this protocol.
    """

    def query(
        self,
        context: str,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
    ) -> QueryResult: ...

    def feedback(
        self,
        query_id: str,
        outcomes: dict[str, str],
        source: str = "unknown",
    ) -> FeedbackResult: ...

    def ingest(
        self,
        source_path: str,
        domain: str,
        source_type: str | None = None,
    ) -> IngestResult: ...

    def domains(self) -> list[DomainInfo]: ...

    def status(self) -> StatusResult: ...

    def explore(
        self,
        node_id: str,
        depth: int = 1,
    ) -> ExploreResult | None: ...

    def rules(
        self,
        domains: list[str] | None = None,
        concept_ids: list[str] | None = None,
        categories: list[str] | None = None,
        include_derived: bool = True,
        min_confidence: float = 0.0,
    ) -> RulesResult: ...

    def ingest_text(
        self,
        text: str,
        domain: str,
        format: str = "text",
        name: str | None = None,
    ) -> IngestResult: ...

    def ingest_structured(
        self,
        concepts: list[dict[str, Any]],
        domain: str,
        edges: list[dict[str, Any]] | None = None,
        rules: list[dict[str, Any]] | None = None,
    ) -> IngestResult: ...

    async def ingest_database(
        self,
        connection_string: str,
        source_id: str,
        domain_map: dict[str, str] | None = None,
        embed_catalog_tables: bool = True,
        extract_rules: bool = True,
    ) -> dict[str, int]: ...


# ---------------------------------------------------------------------------
# LocalQortexClient — direct in-process, no MCP
# ---------------------------------------------------------------------------


class LocalQortexClient:
    """In-process QortexClient backed by VectorIndex + GraphBackend.

    No MCP subprocess, no network. Composes vec and graph layers directly.
    This is the reference implementation — all MCP tools delegate to the
    same logic.

    When mode="graph" (or "auto" with graph available), uses GraphRAGAdapter
    for graph-enhanced retrieval with teleportation factors + online edge gen.
    When mode="vec", uses VecOnlyAdapter (pure cosine similarity).
    """

    def __init__(
        self,
        vector_index: Any,
        backend: Any,
        embedding_model: Any = None,
        llm_backend: Any = None,
        mode: str = "auto",
        interoception: Any = None,
    ) -> None:
        from qortex.hippocampus.adapter import GraphRAGAdapter, VecOnlyAdapter

        self._vector_index = vector_index
        self._backend = backend
        self._embedding_model = embedding_model
        self._llm_backend = llm_backend
        self._mode = mode

        if vector_index is not None and embedding_model is not None:
            if mode == "graph" or (mode == "auto" and self._has_graph_edges()):
                self._adapter = GraphRAGAdapter(
                    vector_index,
                    backend,
                    embedding_model,
                    interoception=interoception,
                )
            else:
                self._adapter = VecOnlyAdapter(vector_index, backend, embedding_model)
        else:
            self._adapter = None

    def _has_graph_edges(self) -> bool:
        """Check if the backend has any edges (worth running PPR)."""
        try:
            from qortex.core.memory import InMemoryBackend

            if isinstance(self._backend, InMemoryBackend):
                return len(self._backend._edges) > 0
        except ImportError:
            pass
        return self._backend.supports_mage()

    def query(
        self,
        context: str,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
    ) -> QueryResult:
        # Clamp to safe ranges (top_k: [1, 1000], min_confidence: [0.0, 1.0])
        top_k = max(1, min(top_k, 1000))
        min_confidence = max(0.0, min(min_confidence, 1.0))

        if self._adapter is None:
            return QueryResult(items=[], query_id=str(uuid.uuid4()))

        result = self._adapter.retrieve(
            query=context, domains=domains, top_k=top_k, min_confidence=min_confidence
        )

        items = [
            QueryItem(
                id=item.id,
                content=item.content,
                score=round(item.score, 4),
                domain=item.domain,
                node_id=item.node_id or item.id,
                metadata=item.metadata,
            )
            for item in result.items
        ]

        # Collect rules linked to activated concepts
        rule_items = self._collect_query_rules(items, domains)

        return QueryResult(items=items, query_id=result.query_id, rules=rule_items)

    def _collect_query_rules(
        self,
        items: list[QueryItem],
        domains: list[str] | None,
    ) -> list[RuleItem]:
        """Collect rules linked to query result concepts."""
        if not items or self._backend is None:
            return []

        from qortex.core.rules import collect_rules_for_concepts

        activated_ids = [item.node_id for item in items]
        scores_map = {item.node_id: item.score for item in items}

        rules = collect_rules_for_concepts(
            self._backend,
            activated_ids,
            domains,
            scores_map,
        )
        return [_rule_to_item(r) for r in rules]

    def explore(
        self,
        node_id: str,
        depth: int = 1,
    ) -> ExploreResult | None:
        """Explore a node's neighborhood in the knowledge graph.

        Returns the node, its typed edges, neighbor nodes, and linked rules.
        Returns None if the node doesn't exist.

        Args:
            node_id: The concept node ID to explore.
            depth: How many hops to traverse (1-3). Default 1 = immediate neighbors.
        """
        depth = max(1, min(depth, 3))

        node = self._backend.get_node(node_id)
        if node is None:
            return None

        # BFS to collect edges and neighbors at each depth
        visited: set[str] = {node_id}
        all_edges: list[EdgeItem] = []
        all_neighbors: list[NodeItem] = []
        frontier: deque[str] = deque([node_id])

        for _hop in range(depth):
            next_frontier: deque[str] = deque()
            while frontier:
                current_id = frontier.popleft()
                edges = list(self._backend.get_edges(current_id, "both"))
                for edge in edges:
                    all_edges.append(_edge_to_item(edge))
                    # Determine neighbor
                    neighbor_id = edge.target_id if edge.source_id == current_id else edge.source_id
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        neighbor_node = self._backend.get_node(neighbor_id)
                        if neighbor_node is not None:
                            all_neighbors.append(_node_to_item(neighbor_node))
                            next_frontier.append(neighbor_id)
            frontier = next_frontier

        # Deduplicate edges by (source_id, target_id, relation_type)
        seen_edges: set[tuple[str, str, str]] = set()
        unique_edges: list[EdgeItem] = []
        for e in all_edges:
            key = (e.source_id, e.target_id, e.relation_type)
            if key not in seen_edges:
                seen_edges.add(key)
                unique_edges.append(e)

        # Collect rules linked to the explored node and its neighbors
        from qortex.core.rules import collect_rules_for_concepts

        all_concept_ids = list(visited)
        rules = collect_rules_for_concepts(self._backend, all_concept_ids)
        rule_items = [_rule_to_item(r) for r in rules]

        return ExploreResult(
            node=_node_to_item(node),
            edges=unique_edges,
            rules=rule_items,
            neighbors=all_neighbors,
        )

    def rules(
        self,
        domains: list[str] | None = None,
        concept_ids: list[str] | None = None,
        categories: list[str] | None = None,
        include_derived: bool = True,
        min_confidence: float = 0.0,
    ) -> RulesResult:
        """Get rules from the knowledge graph via the projector system.

        Delegates to FlatRuleSource.derive() — the projector system, not ad-hoc.

        Args:
            domains: Filter to these domains. None = all.
            concept_ids: If provided, only return rules linked to these concepts.
            categories: Filter by rule category.
            include_derived: Include edge-derived rules (default True).
            min_confidence: Minimum rule confidence.
        """
        from qortex.projectors.models import ProjectionFilter
        from qortex.projectors.sources.flat import FlatRuleSource

        filt = ProjectionFilter(
            domains=domains,
            categories=categories,
            min_confidence=min_confidence,
        )

        source = FlatRuleSource(backend=self._backend, include_derived=include_derived)
        all_rules = source.derive(domains=domains, filters=filt)

        # Filter by concept_ids if provided
        if concept_ids is not None:
            concept_set = set(concept_ids)
            all_rules = [r for r in all_rules if concept_set.intersection(r.source_concepts)]

        # Count distinct domains
        domain_names = {r.domain for r in all_rules}

        rule_items = [_rule_to_item(r) for r in all_rules]

        return RulesResult(
            rules=rule_items,
            domain_count=len(domain_names),
        )

    def feedback(
        self,
        query_id: str,
        outcomes: dict[str, str],
        source: str = "unknown",
    ) -> FeedbackResult:
        if self._adapter is not None:
            self._adapter.feedback(query_id, outcomes)

        return FeedbackResult(
            status="recorded",
            query_id=query_id,
            outcome_count=len(outcomes),
            source=source,
        )

    def ingest(
        self,
        source_path: str,
        domain: str,
        source_type: str | None = None,
    ) -> IngestResult:
        from pathlib import Path

        from qortex.ingest.base import Source

        path = Path(source_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {source_path}")

        allowed = {"text", "markdown", "pdf"}
        if source_type is not None and source_type not in allowed:
            raise ValueError(f"Invalid source_type: {source_type}. Must be one of {allowed}")

        if source_type is None:
            ext = path.suffix.lower()
            type_map = {".md": "markdown", ".markdown": "markdown", ".pdf": "pdf", ".txt": "text"}
            source_type = type_map.get(ext, "text")

        source = Source(path=path, source_type=source_type, name=path.name)
        llm = self._get_llm_backend()

        if source_type == "markdown":
            from qortex.ingest.markdown import MarkdownIngestor

            ingestor = MarkdownIngestor(llm, embedding_model=self._embedding_model)
        elif source_type == "pdf":
            from qortex.ingest.pdf import PDFIngestor

            ingestor = PDFIngestor(llm, embedding_model=self._embedding_model)
        else:
            from qortex.ingest.text import TextIngestor

            ingestor = TextIngestor(llm, embedding_model=self._embedding_model)

        manifest = ingestor.ingest(source, domain=domain)
        self._backend.ingest_manifest(manifest)
        self._index_manifest_embeddings(manifest)
        self._maybe_upgrade_adapter()

        return IngestResult(
            domain=domain,
            source=path.name,
            concepts=len(manifest.concepts),
            edges=len(manifest.edges),
            rules=len(manifest.rules),
            warnings=manifest.warnings,
        )

    def ingest_text(
        self,
        text: str,
        domain: str,
        format: str = "text",
        name: str | None = None,
    ) -> IngestResult:
        """Ingest raw text or markdown into a domain.

        Creates a Source with raw_content and runs the appropriate ingestor.
        No file needed — paste text directly.

        Args:
            text: The text content to ingest.
            domain: Target domain name.
            format: "text" or "markdown". Selects the ingestor.
            name: Optional human-readable source name.
        """
        if format not in ("text", "markdown"):
            raise ValueError(f"Invalid format: {format}. Must be 'text' or 'markdown'")

        if not text or not text.strip():
            return IngestResult(
                domain=domain,
                source=name or "raw_text",
                concepts=0,
                edges=0,
                rules=0,
                warnings=["Empty text provided"],
            )

        from qortex.ingest.base import Source

        source = Source(
            raw_content=text,
            source_type=format,
            name=name or "raw_text",
        )

        llm = self._get_llm_backend()

        if format == "markdown":
            from qortex.ingest.markdown import MarkdownIngestor

            ingestor = MarkdownIngestor(llm, embedding_model=self._embedding_model)
        else:
            from qortex.ingest.text import TextIngestor

            ingestor = TextIngestor(llm, embedding_model=self._embedding_model)

        manifest = ingestor.ingest(source, domain=domain)
        self._backend.ingest_manifest(manifest)
        self._index_manifest_embeddings(manifest)
        self._maybe_upgrade_adapter()

        return IngestResult(
            domain=domain,
            source=name or "raw_text",
            concepts=len(manifest.concepts),
            edges=len(manifest.edges),
            rules=len(manifest.rules),
            warnings=manifest.warnings,
        )

    def ingest_structured(
        self,
        concepts: list[dict[str, Any]],
        domain: str,
        edges: list[dict[str, Any]] | None = None,
        rules: list[dict[str, Any]] | None = None,
    ) -> IngestResult:
        """Ingest pre-structured data directly into the knowledge graph.

        Bypasses LLM extraction — takes concepts, edges, and rules directly.
        Each concept dict must have 'name' and 'description'.
        Each edge dict must have 'source' and 'target' (concept names or IDs)
        and 'relation_type' (a RelationType value).
        Each rule dict must have 'text'.

        Args:
            concepts: List of concept dicts with at minimum {name, description}.
            domain: Target domain name.
            edges: Optional list of edge dicts {source, target, relation_type}.
            rules: Optional list of rule dicts {text, category?}.
        """
        import hashlib

        from qortex.core.models import (
            ConceptEdge,
            ConceptNode,
            ExplicitRule,
            IngestionManifest,
            RelationType,
            SourceMetadata,
        )

        edges = edges or []
        rules = rules or []

        if self._backend.get_domain(domain) is None:
            self._backend.create_domain(domain)

        source_id = f"structured:{hashlib.sha256(domain.encode()).hexdigest()[:12]}"

        # Build ConceptNodes
        concept_nodes: list[ConceptNode] = []
        name_to_id: dict[str, str] = {}
        for c in concepts:
            name = c["name"]
            desc = c.get("description", name)
            node_id = c.get("id", f"{domain}:{hashlib.sha256(name.encode()).hexdigest()[:12]}")
            node = ConceptNode(
                id=node_id,
                name=name,
                description=desc,
                domain=domain,
                source_id=source_id,
                properties=c.get("properties", {}),
                confidence=c.get("confidence", 1.0),
            )
            concept_nodes.append(node)
            name_to_id[name] = node_id
            name_to_id[node_id] = node_id  # allow lookup by ID too

        # Generate embeddings
        if self._embedding_model is not None:
            texts = [f"{n.name}: {n.description}" for n in concept_nodes]
            embeddings = self._embedding_model.embed(texts)
            for node, emb in zip(concept_nodes, embeddings):
                node.embedding = emb

        # Build ConceptEdges
        concept_edges: list[ConceptEdge] = []
        for e in edges:
            source_ref = e["source"]
            target_ref = e["target"]
            rel_type_str = e["relation_type"]

            # Resolve names to IDs
            source_node_id = name_to_id.get(source_ref)
            target_node_id = name_to_id.get(target_ref)
            if source_node_id is None or target_node_id is None:
                continue

            # Validate relation type
            try:
                rel_type = RelationType(rel_type_str)
            except ValueError:
                continue

            concept_edges.append(
                ConceptEdge(
                    source_id=source_node_id,
                    target_id=target_node_id,
                    relation_type=rel_type,
                    confidence=e.get("confidence", 1.0),
                )
            )

        # Build ExplicitRules
        explicit_rules: list[ExplicitRule] = []
        for r in rules:
            rule_id = f"{domain}:rule:{hashlib.sha256(r['text'].encode()).hexdigest()[:12]}"
            explicit_rules.append(
                ExplicitRule(
                    id=rule_id,
                    text=r["text"],
                    domain=domain,
                    source_id=source_id,
                    category=r.get("category"),
                    confidence=r.get("confidence", 1.0),
                    concept_ids=[n.id for n in concept_nodes],
                )
            )

        source_meta = SourceMetadata(
            id=source_id,
            name="structured_input",
            source_type="text",
            path_or_url="structured://input",
        )

        manifest = IngestionManifest(
            source=source_meta,
            domain=domain,
            concepts=concept_nodes,
            edges=concept_edges,
            rules=explicit_rules,
        )

        self._backend.ingest_manifest(manifest)
        self._index_manifest_embeddings(manifest)
        self._maybe_upgrade_adapter()

        return IngestResult(
            domain=domain,
            source="structured_input",
            concepts=len(concept_nodes),
            edges=len(concept_edges),
            rules=len(explicit_rules),
        )

    def _index_manifest_embeddings(self, manifest: Any) -> None:
        """Dual-write embeddings from a manifest to backend + vector index."""
        ids_with_embeddings = []
        embeddings_list = []
        for concept in manifest.concepts:
            if concept.embedding is not None:
                self._backend.add_embedding(concept.id, concept.embedding)
                ids_with_embeddings.append(concept.id)
                embeddings_list.append(concept.embedding)

        if self._vector_index is not None and ids_with_embeddings:
            self._vector_index.add(ids_with_embeddings, embeddings_list)

    def _maybe_upgrade_adapter(self) -> None:
        """If mode=auto and currently VecOnly, upgrade to GraphRAG when edges exist."""
        from qortex.hippocampus.adapter import GraphRAGAdapter, VecOnlyAdapter

        if self._mode != "auto":
            return
        if self._adapter is None:
            return
        if isinstance(self._adapter, GraphRAGAdapter):
            return
        if not isinstance(self._adapter, VecOnlyAdapter):
            return
        if not self._has_graph_edges():
            return

        self._adapter = GraphRAGAdapter(
            self._vector_index,
            self._backend,
            self._embedding_model,
        )

    # -- Public accessors for adapter interop --

    @property
    def embedding_model(self) -> Any:
        """The embedding model, or None. Public accessor for adapters."""
        return self._embedding_model

    def add_concepts(
        self,
        texts: list[str],
        domain: str = "default",
        metadatas: list[dict[str, Any]] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        """Add text concepts to the knowledge graph.

        Creates ConceptNodes, generates embeddings, and indexes them.
        This is the text-based counterpart to ingest() (file-based).

        Args:
            texts: Texts to add as concepts.
            domain: Domain to add concepts to.
            metadatas: Optional metadata per text.
            ids: Optional IDs. Auto-generated if not provided.

        Returns:
            List of concept IDs added.
        """
        import hashlib

        from qortex.core.models import ConceptNode

        metadatas_list = metadatas or [{}] * len(texts)

        if ids is None:
            ids = [f"{domain}:{hashlib.sha256(t.encode()).hexdigest()[:12]}" for t in texts]

        if self._backend.get_domain(domain) is None:
            self._backend.create_domain(domain)

        embeddings = self._embedding_model.embed(texts)
        for text, meta, node_id, emb in zip(texts, metadatas_list, ids, embeddings):
            name = meta.get("name", text[:80])
            node = ConceptNode(
                id=node_id,
                name=name,
                description=text,
                domain=domain,
                source_id=meta.get("source_id", "langchain"),
                properties=meta,
            )
            self._backend.add_node(node)
            self._backend.add_embedding(node_id, emb)

        if self._vector_index is not None:
            self._vector_index.add(ids, embeddings)

        return ids

    def get_nodes(self, ids: list[str]) -> list[NodeItem]:
        """Get nodes by their IDs. Public accessor for adapters.

        Returns:
            List of NodeItems for found IDs (missing IDs are skipped).
        """
        result = []
        for node_id in ids:
            node = self._backend.get_node(node_id)
            if node is not None:
                result.append(_node_to_item(node))
        return result

    def domains(self) -> list[DomainInfo]:
        return [
            DomainInfo(
                name=d.name,
                description=d.description,
                concept_count=d.concept_count,
                edge_count=d.edge_count,
                rule_count=d.rule_count,
            )
            for d in self._backend.list_domains()
        ]

    def status(self) -> StatusResult:
        return StatusResult(
            status="ok",
            backend=type(self._backend).__name__,
            vector_index=type(self._vector_index).__name__ if self._vector_index else None,
            vector_search=self._vector_index is not None,
            graph_algorithms=self._backend.supports_mage() if self._backend else False,
            domain_count=len(self._backend.list_domains()) if self._backend else 0,
            embedding_model=(
                getattr(self._embedding_model, "_model_name", type(self._embedding_model).__name__)
                if self._embedding_model
                else None
            ),
        )

    async def connect_source(
        self,
        config: Any,
        ingest: Any = None,
    ) -> dict:
        """Connect to a database source and discover its schema.

        Args:
            config: SourceConfig for the database connection.
            ingest: Optional IngestConfig.

        Returns:
            Dict with source_id, table count, table names.
        """
        from qortex.sources.base import SourceConfig

        if not isinstance(config, SourceConfig):
            raise TypeError(f"Expected SourceConfig, got {type(config).__name__}")

        try:
            from qortex.sources.postgres import PostgresSourceAdapter
        except ImportError:
            raise ImportError("asyncpg required. Install qortex[source-postgres].")

        adapter = PostgresSourceAdapter()
        await adapter.connect(config)
        schemas = await adapter.discover()

        # Store for later sync
        if not hasattr(self, "_source_adapters"):
            self._source_adapters: dict[str, Any] = {}
            self._source_configs: dict[str, Any] = {}
            self._source_schemas: dict[str, list] = {}

        self._source_adapters[config.source_id] = adapter
        self._source_configs[config.source_id] = config
        self._source_schemas[config.source_id] = schemas

        return {
            "source_id": config.source_id,
            "tables": len(schemas),
            "table_names": [t.name for t in schemas],
        }

    async def sync_source(
        self,
        source_id: str,
        tables: list[str] | None = None,
    ) -> Any:
        """Sync a connected source's data to the vec layer.

        Args:
            source_id: The source to sync.
            tables: Tables to sync. None = all.

        Returns:
            SyncResult with sync statistics.
        """
        if not hasattr(self, "_source_adapters") or source_id not in self._source_adapters:
            raise ValueError(f"Source '{source_id}' not connected. Call connect_source() first.")

        adapter = self._source_adapters[source_id]
        return await adapter.sync(
            tables=tables,
            vector_index=self._vector_index,
            embedding_model=self._embedding_model,
        )

    def _get_llm_backend(self):
        if self._llm_backend is not None:
            return self._llm_backend

        try:
            from qortex.ingest.backends.anthropic import AnthropicBackend

            self._llm_backend = AnthropicBackend()
            return self._llm_backend
        except ImportError:
            logger.debug("anthropic backend not installed, falling back to StubLLMBackend")
        except Exception:
            logger.warning(
                "Failed to initialize AnthropicBackend, falling back to StubLLMBackend",
                exc_info=True,
            )

        from qortex.ingest.base import StubLLMBackend

        self._llm_backend = StubLLMBackend()
        return self._llm_backend

    async def ingest_database(
        self,
        connection_string: str,
        source_id: str,
        domain_map: dict[str, str] | None = None,
        embed_catalog_tables: bool = True,
        extract_rules: bool = True,
    ) -> dict[str, int]:
        """Ingest a PostgreSQL database schema into the knowledge graph.

        Connects via asyncpg, discovers schema with FK/CHECK/UNIQUE metadata,
        maps to graph structure, and creates ConceptNodes/Edges/Rules.

        Returns: {concepts: int, edges: int, rules: int}
        """
        from dataclasses import dataclass as _dc
        from dataclasses import field as _field

        @_dc
        class _IngestConf:
            embed_catalog_tables: bool = embed_catalog_tables
            extract_rules: bool = extract_rules

        @_dc
        class _SourceConf:
            source_id: str = source_id
            schemas: list[str] = _field(default_factory=lambda: ["public"])
            domain_map: dict[str, str] = _field(default_factory=dict)

        import asyncpg

        from qortex.sources.postgres_graph import PostgresGraphIngestor

        source_conf = _SourceConf()
        source_conf.domain_map = domain_map or {}

        ingestor = PostgresGraphIngestor(
            config=source_conf,
            ingest_config=_IngestConf(),
            backend=self._backend,
            embedding_model=self._embedding_model,
        )

        conn = await asyncpg.connect(connection_string)
        try:
            return await ingestor.run(conn=conn)
        finally:
            await conn.close()
