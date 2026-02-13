"""Agno adapter: qortex as a knowledge source for Agno agents.

Implements the full agno KnowledgeProtocol:
    build_context()  — system prompt instructions
    get_tools()      — search + explore + feedback tools for the agent
    aget_tools()     — async variant (same tools)
    retrieve()       — document retrieval for context injection
    aretrieve()      — async variant

Usage:
    from qortex.client import LocalQortexClient
    from qortex.adapters.agno import QortexKnowledge

    client = LocalQortexClient(vector_index, backend, embedding_model)
    knowledge = QortexKnowledge(client=client, domains=["security"])

    # Drop-in replacement for agno Knowledge
    agent = Agent(knowledge=knowledge, search_knowledge=True)

Does NOT require agno to be installed — returns plain dicts matching
agno's Document shape when agno is not available.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from qortex.client import QortexClient

logger = logging.getLogger(__name__)


def _to_document(item: Any) -> Any:
    """Convert a QueryItem to an agno Document or compatible dict."""
    return item.to_agno_document()


def _docs_to_string(docs: list[Any], agent: Any = None) -> str:
    """Convert documents to a formatted string for the LLM."""
    if agent is not None and hasattr(agent, "_convert_documents_to_string"):
        dict_docs = [doc.to_dict() if hasattr(doc, "to_dict") else doc for doc in docs]
        return agent._convert_documents_to_string(dict_docs)

    if not docs:
        return "No documents found"

    parts = []
    for i, doc in enumerate(docs, 1):
        content = doc.content if hasattr(doc, "content") else doc.get("content", "")
        score = (
            doc.reranking_score
            if hasattr(doc, "reranking_score")
            else doc.get("reranking_score", "")
        )
        parts.append(f"[{i}] (score={score}) {content}")
    return "\n---\n".join(parts)


class QortexKnowledge:
    """Qortex as an Agno knowledge source.

    Implements agno's KnowledgeProtocol so it works as a drop-in replacement
    for Agent(knowledge=...).

    The agent gets three tools:
      - search_knowledge_base(query) — semantic + graph search
      - explore_knowledge_graph(node_id, depth) — BFS graph exploration
      - report_knowledge_feedback(item_id, outcome) — closes the learning loop

    The graph exploration and feedback tools are what vanilla agno Knowledge
    can't do. The feedback tool enables online Thompson Sampling — retrieval
    quality improves over time without retraining.
    """

    _SEARCH_INSTRUCTIONS = (
        "Use `search_knowledge_base` to search the knowledge graph for relevant information. "
        "The knowledge graph combines vector similarity with graph-based PageRank, so results "
        "capture both semantic relevance and structural importance.\n"
        "After using results, call `report_knowledge_feedback` with the item IDs and outcomes "
        '("accepted", "rejected", "partial") to improve future retrievals.\n'
        "Use `explore_knowledge_graph` to traverse the graph from a specific node when you "
        "need to understand relationships and context around a concept."
    )

    def __init__(
        self,
        client: QortexClient,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
        feedback_source: str = "agno",
        enable_explore: bool = True,
        enable_feedback: bool = True,
    ) -> None:
        self._client = client
        self._domains = domains
        self._top_k = top_k
        self._min_confidence = min_confidence
        self._feedback_source = feedback_source
        self._enable_explore = enable_explore
        self._enable_feedback = enable_feedback
        self._last_query_id: str | None = None

    # ------------------------------------------------------------------
    # KnowledgeProtocol: build_context
    # ------------------------------------------------------------------

    def build_context(self, **kwargs: Any) -> str:
        """Build context string for the agent's system prompt."""
        parts = [self._SEARCH_INSTRUCTIONS]
        if self._domains:
            parts.append(f"Knowledge domains available: {', '.join(self._domains)}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # KnowledgeProtocol: get_tools / aget_tools
    # ------------------------------------------------------------------

    def get_tools(
        self,
        run_response: Any = None,
        run_context: Any = None,
        knowledge_filters: Any = None,
        async_mode: bool = False,
        enable_agentic_filters: bool = False,
        agent: Any = None,
        **kwargs: Any,
    ) -> list[Callable]:
        """Get tools to expose to the agent.

        Returns search, explore, and feedback tools that the agent can call.
        """
        tools: list[Callable] = []

        # --- search tool ---
        search_fn = self._make_search_tool(run_response, agent)
        tools.append(search_fn)

        # --- explore tool ---
        if self._enable_explore:
            tools.append(self._make_explore_tool())

        # --- feedback tool ---
        if self._enable_feedback:
            tools.append(self._make_feedback_tool())

        return tools

    async def aget_tools(
        self,
        run_response: Any = None,
        run_context: Any = None,
        knowledge_filters: Any = None,
        async_mode: bool = True,
        enable_agentic_filters: bool = False,
        agent: Any = None,
        **kwargs: Any,
    ) -> list[Callable]:
        """Async version of get_tools."""
        return self.get_tools(
            run_response=run_response,
            run_context=run_context,
            knowledge_filters=knowledge_filters,
            async_mode=async_mode,
            enable_agentic_filters=enable_agentic_filters,
            agent=agent,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # KnowledgeProtocol: retrieve / aretrieve
    # ------------------------------------------------------------------

    def retrieve(self, query: str, **kwargs: Any) -> list[Any]:
        """Retrieve relevant documents for a query.

        Returns:
            List of agno Documents (or dicts with same shape).
        """
        top_k = kwargs.get("top_k", self._top_k)
        min_confidence = kwargs.get("min_confidence", self._min_confidence)

        result = self._client.query(
            context=query,
            domains=self._domains,
            top_k=top_k,
            min_confidence=min_confidence,
        )
        self._last_query_id = result.query_id

        return [_to_document(item) for item in result.items]

    async def aretrieve(self, query: str, **kwargs: Any) -> list[Any]:
        """Async version of retrieve. Delegates to sync (qortex client is sync)."""
        return self.retrieve(query, **kwargs)

    # ------------------------------------------------------------------
    # Public: feedback (also exposed as a tool)
    # ------------------------------------------------------------------

    def feedback(self, outcomes: dict[str, str]) -> None:
        """Report feedback for the last retrieval. Closes the learning loop."""
        if self._last_query_id is None:
            return
        self._client.feedback(
            query_id=self._last_query_id,
            outcomes=outcomes,
            source=self._feedback_source,
        )

    @property
    def last_query_id(self) -> str | None:
        return self._last_query_id

    # ------------------------------------------------------------------
    # Tool factories
    # ------------------------------------------------------------------

    def _make_search_tool(self, run_response: Any = None, agent: Any = None) -> Callable:
        """Create the search_knowledge_base tool."""
        client = self._client
        domains = self._domains
        top_k = self._top_k
        min_confidence = self._min_confidence
        # Capture self for query_id tracking
        knowledge = self

        def search_knowledge_base(query: str, num_documents: int | None = None) -> str:
            """Search the knowledge graph for information relevant to a query.

            Uses vector similarity + graph-based PageRank for retrieval.
            Results include relevance scores combining semantic and structural signals.

            Args:
                query: The search query.
                num_documents: Max results to return (default: configured top_k).

            Returns:
                Formatted search results with scores and content.
            """
            k = num_documents if num_documents is not None else top_k

            result = client.query(
                context=query,
                domains=domains,
                top_k=k,
                min_confidence=min_confidence,
            )
            knowledge._last_query_id = result.query_id

            docs = [_to_document(item) for item in result.items]

            # Attach references to run_response if available
            if run_response is not None and docs:
                try:
                    from agno.models.message import MessageReferences

                    references = MessageReferences(
                        query=query,
                        references=[
                            doc.to_dict() if hasattr(doc, "to_dict") else doc for doc in docs
                        ],
                    )
                    if run_response.references is None:
                        run_response.references = []
                    run_response.references.append(references)
                except ImportError:
                    pass

            if not docs:
                return "No documents found in the knowledge graph."

            # Append rules if any
            rules_text = ""
            if result.rules:
                rules_parts = [f"- {r.text} (confidence={r.confidence:.2f})" for r in result.rules]
                rules_text = "\n\nRelevant rules:\n" + "\n".join(rules_parts)

            return _docs_to_string(docs, agent) + rules_text

        return search_knowledge_base

    def _make_explore_tool(self) -> Callable:
        """Create the explore_knowledge_graph tool."""
        client = self._client

        def explore_knowledge_graph(node_id: str, depth: int = 1) -> str:
            """Explore a node's neighborhood in the knowledge graph.

            Traverses the graph via BFS to show relationships, neighbor concepts,
            and linked rules around a specific node.

            Args:
                node_id: The concept node ID to explore (from search results).
                depth: How many hops to traverse (1-3). Default 1.

            Returns:
                JSON description of the node, its edges, neighbors, and rules.
            """
            result = client.explore(node_id=node_id, depth=depth)
            if result is None:
                return f"Node '{node_id}' not found in the knowledge graph."

            output = {
                "node": {
                    "id": result.node.id,
                    "name": result.node.name,
                    "description": result.node.description,
                    "domain": result.node.domain,
                },
                "edges": [
                    {
                        "source": e.source_id,
                        "target": e.target_id,
                        "relation": e.relation_type,
                        "confidence": round(e.confidence, 3),
                    }
                    for e in result.edges
                ],
                "neighbors": [
                    {"id": n.id, "name": n.name, "domain": n.domain} for n in result.neighbors
                ],
                "rules": [
                    {"text": r.text, "confidence": round(r.confidence, 3)} for r in result.rules
                ],
            }
            return json.dumps(output, indent=2)

        return explore_knowledge_graph

    def _make_feedback_tool(self) -> Callable:
        """Create the report_knowledge_feedback tool."""
        knowledge = self

        def report_knowledge_feedback(item_id: str, outcome: str) -> str:
            """Report feedback on a knowledge item to improve future retrievals.

            Call this after using knowledge items. The feedback updates the
            Thompson Sampling posterior, so future queries rank better items
            higher over time.

            Args:
                item_id: The ID of the knowledge item (from search results).
                outcome: One of "accepted", "rejected", or "partial".

            Returns:
                Confirmation of recorded feedback.
            """
            if outcome not in ("accepted", "rejected", "partial"):
                return f"Invalid outcome '{outcome}'. Use: accepted, rejected, partial."

            if knowledge._last_query_id is None:
                return "No recent query to provide feedback for."

            knowledge._client.feedback(
                query_id=knowledge._last_query_id,
                outcomes={item_id: outcome},
                source=knowledge._feedback_source,
            )
            return f"Feedback recorded: {item_id} → {outcome}"

        return report_knowledge_feedback
