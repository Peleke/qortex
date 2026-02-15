"""Retrieval pipeline for LongMemEval.

Given a benchmark question, queries the qortex knowledge graph for
relevant context and formats it for LLM answer generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from qortex.client import LocalQortexClient, QueryResult

logger = logging.getLogger(__name__)


@dataclass
class RetrievalContext:
    """Context retrieved from qortex for answering a question."""

    context_text: str
    query_id: str
    items: list[dict[str, Any]] = field(default_factory=list)
    rules: list[dict[str, Any]] = field(default_factory=list)
    item_count: int = 0


def retrieve_context(
    client: LocalQortexClient,
    question: str,
    domain: str = "longmemeval",
    top_k: int = 20,
    question_date: str | None = None,
) -> RetrievalContext:
    """Query qortex for context relevant to a benchmark question.

    Uses the full GraphRAG pipeline: vec search -> online edge gen ->
    PPR -> combined scoring. Returns formatted context string suitable
    for injection into an LLM prompt.

    Args:
        client: LocalQortexClient instance.
        question: The benchmark question text.
        domain: Domain to query within.
        top_k: Number of results to retrieve.
        question_date: Optional date context for temporal queries.

    Returns:
        RetrievalContext with formatted text and metadata.
    """
    # Augment query with date context for temporal reasoning
    query_text = question
    if question_date:
        query_text = f"[Current date: {question_date}] {question}"

    result: QueryResult = client.query(
        context=query_text,
        domains=[domain],
        top_k=top_k,
    )

    # Format context from retrieved items
    context_parts: list[str] = []

    for item in result.items:
        # Extract temporal metadata if present
        meta = item.metadata or {}
        session_date = meta.get("session_date", "")
        role = meta.get("role", "")

        if session_date:
            prefix = f"[{session_date}]"
        else:
            prefix = ""

        if role:
            prefix = f"{prefix} {role}:"

        content = item.content
        if prefix:
            context_parts.append(f"{prefix} {content}")
        else:
            context_parts.append(content)

    # Include rules as additional context
    rule_parts: list[str] = []
    for rule in result.rules:
        rule_parts.append(f"- {rule.text}")

    context_text = "\n\n".join(context_parts)
    if rule_parts:
        context_text += "\n\nRelevant rules:\n" + "\n".join(rule_parts)

    return RetrievalContext(
        context_text=context_text,
        query_id=result.query_id,
        items=[
            {
                "id": item.id,
                "node_id": item.node_id,
                "content": item.content,
                "score": item.score,
                "domain": item.domain,
                "metadata": item.metadata,
            }
            for item in result.items
        ],
        rules=[
            {"id": rule.id, "text": rule.text, "domain": rule.domain}
            for rule in result.rules
        ],
        item_count=len(result.items),
    )


def format_answer_prompt(
    question: str,
    context_text: str,
    question_date: str | None = None,
) -> str:
    """Format the LLM prompt for answer generation.

    Args:
        question: The benchmark question.
        context_text: Formatted context string from retrieve_context().
        question_date: Optional date for temporal grounding.

    Returns:
        Formatted prompt string.
    """
    parts = []

    if question_date:
        parts.append(f"Current date: {question_date}")

    parts.append(
        "You are a helpful assistant with access to the user's conversation history. "
        "Use the following retrieved context to answer the question. "
        "If the context does not contain enough information to answer, say so. "
        "Be concise and direct."
    )

    if context_text:
        parts.append(f"Retrieved context:\n{context_text}")
    else:
        parts.append("No relevant context was found in the conversation history.")

    parts.append(f"Question: {question}")
    parts.append("Answer:")

    return "\n\n".join(parts)
