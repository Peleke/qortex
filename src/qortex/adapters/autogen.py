"""AutoGen (AG2) adapter: Memory implementation backed by qortex.

Usage:
    from qortex.client import LocalQortexClient
    from qortex.adapters.autogen import QortexMemory

    client = LocalQortexClient(vector_index, backend, embedding_model)
    memory = QortexMemory(client=client, domains=["security"])

    # Works anywhere autogen expects a Memory
    result = await memory.query("What is OAuth2?")
    # → MemoryQueryResult(results=[MemoryContent(...), ...])

    # In an agent pipeline:
    await memory.update_context(model_context)

Implements AutoGen's Memory interface (5 async methods):
    update_context, query, add, clear, close

Does NOT require autogen to be installed — returns compatible types
matching AutoGen's MemoryContent/MemoryQueryResult shapes. When autogen-core
IS installed, returns actual autogen types for full compatibility.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from qortex.client import QortexClient

# ---------------------------------------------------------------------------
# Conditional autogen imports — fall back to compatible plain types
# ---------------------------------------------------------------------------

try:
    from autogen_core.memory import (
        MemoryContent,
        MemoryMimeType,
        MemoryQueryResult,
        UpdateContextResult,
    )

    _HAS_AUTOGEN = True
except ImportError:
    _HAS_AUTOGEN = False


def _make_memory_content(
    content: str,
    score: float,
    domain: str,
    node_id: str,
    item_id: str,
    extra_meta: dict[str, Any] | None = None,
) -> Any:
    """Build a MemoryContent (real or dict fallback)."""
    meta = {"score": score, "domain": domain, "node_id": node_id, "id": item_id}
    if extra_meta:
        meta.update(extra_meta)

    if _HAS_AUTOGEN:
        return MemoryContent(
            content=content,
            mime_type=MemoryMimeType.TEXT,
            metadata=meta,
        )
    return {"content": content, "mime_type": "text/plain", "metadata": meta}


def _make_query_result(results: list[Any]) -> Any:
    """Build a MemoryQueryResult (real or dict fallback)."""
    if _HAS_AUTOGEN:
        return MemoryQueryResult(results=results)
    return {"results": results}


def _make_update_result(query_result: Any) -> Any:
    """Build an UpdateContextResult (real or dict fallback)."""
    if _HAS_AUTOGEN:
        return UpdateContextResult(memories=query_result)
    return {"memories": query_result}


# ---------------------------------------------------------------------------
# Config (plain dataclass — no pydantic dependency for the adapter itself)
# ---------------------------------------------------------------------------


@dataclass
class QortexMemoryConfig:
    """Configuration for QortexMemory.

    When autogen-core is installed, this can be serialized via the
    Component protocol (_to_config / _from_config).
    """

    domains: list[str] | None = None
    top_k: int = 5
    score_threshold: float = 0.0
    feedback_source: str = "autogen"


# ---------------------------------------------------------------------------
# QortexMemory — AutoGen Memory interface backed by qortex
# ---------------------------------------------------------------------------


class QortexMemory:
    """Drop-in replacement for autogen's Memory.

    Backed by QortexClient instead of ChromaDB. Implements all 5 async
    Memory methods: update_context, query, add, clear, close.

    All methods are async. The underlying QortexClient is synchronous,
    so blocking calls are wrapped in asyncio.to_thread() to avoid
    blocking the event loop.

    When autogen-core is installed, returns actual autogen types
    (MemoryContent, MemoryQueryResult, UpdateContextResult).
    When not installed, returns dicts with the same shape.
    """

    component_type = "memory"

    def __init__(
        self,
        client: QortexClient,
        config: QortexMemoryConfig | None = None,
        *,
        domains: list[str] | None = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        feedback_source: str = "autogen",
    ) -> None:
        self._client = client
        self._config = config or QortexMemoryConfig(
            domains=domains,
            top_k=top_k,
            score_threshold=score_threshold,
            feedback_source=feedback_source,
        )
        self._last_query_id: str | None = None

    # -- Memory interface methods --

    async def update_context(self, model_context: Any) -> Any:
        """Extract the last user message from model_context, query memory,
        and inject results as a system message.

        Follows the same pattern as AutoGen's ChromaDBVectorMemory:
        1. Get the last message from context
        2. Query qortex for relevant knowledge
        3. Create a system message with the results
        4. Add the system message to the context
        5. Return UpdateContextResult

        Args:
            model_context: An AutoGen ChatCompletionContext, or any object
                with an async get_messages() method and add_message() method.

        Returns:
            UpdateContextResult (or compatible dict).
        """
        # Extract the last user message
        query_text = await self._extract_last_message(model_context)
        if not query_text:
            empty = _make_query_result([])
            return _make_update_result(empty)

        # Query qortex
        query_result = await self.query(query_text)
        results = (
            query_result.results
            if hasattr(query_result, "results")
            else query_result.get("results", [])
        )

        # Inject as system message if we got results
        if results:
            memory_strings = []
            for mc in results:
                content = mc.content if hasattr(mc, "content") else mc.get("content", "")
                memory_strings.append(str(content))

            memory_text = "Relevant knowledge from memory:\n" + "\n---\n".join(memory_strings)

            # Try to inject via autogen's message types
            try:
                from autogen_core.models import SystemMessage

                await model_context.add_message(SystemMessage(content=memory_text))
            except (ImportError, AttributeError):
                # If autogen isn't available or model_context doesn't support
                # add_message, skip injection (query results still returned)
                pass

        return _make_update_result(query_result)

    async def query(
        self,
        query: Any,
        cancellation_token: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Query qortex for relevant knowledge.

        Args:
            query: A search string, or a MemoryContent object.
            cancellation_token: AutoGen CancellationToken (accepted but unused).
            **kwargs: Additional arguments (top_k, score_threshold overrides).

        Returns:
            MemoryQueryResult with list of MemoryContent results.
        """
        # Extract query text
        if isinstance(query, str):
            query_text = query
        elif hasattr(query, "content"):
            query_text = str(query.content)
        elif isinstance(query, dict):
            query_text = str(query.get("content", query))
        else:
            query_text = str(query)

        top_k = kwargs.get("top_k", self._config.top_k)
        score_threshold = kwargs.get("score_threshold", self._config.score_threshold)

        # Run sync client.query in a thread to avoid blocking
        result = await asyncio.to_thread(
            self._client.query,
            context=query_text,
            domains=self._config.domains,
            top_k=top_k,
            min_confidence=score_threshold,
        )
        self._last_query_id = result.query_id

        # Convert to MemoryContent list
        memory_contents = [
            _make_memory_content(
                content=item.content,
                score=item.score,
                domain=item.domain,
                node_id=item.node_id,
                item_id=item.id,
                extra_meta=item.metadata,
            )
            for item in result.items
        ]

        return _make_query_result(memory_contents)

    async def add(
        self,
        content: Any,
        cancellation_token: Any | None = None,
    ) -> None:
        """Add content to qortex's knowledge graph.

        Ingests text via client.ingest_text(). For structured data,
        use client.ingest_structured() directly.

        Args:
            content: A MemoryContent object, dict, or string.
            cancellation_token: AutoGen CancellationToken (accepted but unused).
        """
        # Extract text from content
        if isinstance(content, str):
            text = content
        elif hasattr(content, "content"):
            text = str(content.content)
        elif isinstance(content, dict):
            text = str(content.get("content", content))
        else:
            text = str(content)

        if not text.strip():
            return

        domain = self._config.domains[0] if self._config.domains else "default"

        await asyncio.to_thread(
            self._client.ingest_text,
            text=text,
            domain=domain,
        )

    async def clear(self) -> None:
        """Clear memory. No-op for qortex (persistent knowledge graph).

        Unlike ChromaDB where you can delete a collection, qortex's
        knowledge graph is persistent and shared. Clearing would destroy
        the entire graph, which is almost never what you want in an
        agent pipeline.

        If you need to reset for testing, create a new client with a
        fresh InMemoryBackend.
        """
        pass

    async def close(self) -> None:
        """Close the memory. No-op for local client (no connection to close)."""
        pass

    # -- Feedback (qortex-specific, not part of AutoGen Memory ABC) --

    async def feedback(self, outcomes: dict[str, str]) -> None:
        """Report feedback for the last query. Closes the learning loop.

        This is a qortex extension — AutoGen's Memory ABC doesn't have
        a feedback method, but the learning loop is what differentiates
        qortex from static vector stores.

        Args:
            outcomes: Dict of {concept_id: "accepted"|"rejected"|"ignored"}.
        """
        if self._last_query_id is None:
            return
        await asyncio.to_thread(
            self._client.feedback,
            query_id=self._last_query_id,
            outcomes=outcomes,
            source=self._config.feedback_source,
        )

    # -- Config / serialization --

    def _to_config(self) -> QortexMemoryConfig:
        """Serialize to config (Component protocol)."""
        return self._config

    @classmethod
    def _from_config(cls, config: QortexMemoryConfig, client: QortexClient) -> QortexMemory:
        """Deserialize from config (Component protocol).

        Note: Unlike autogen's built-in memories, this requires a client
        instance because QortexClient manages the vector index and graph
        backend, which can't be serialized as a simple config.
        """
        return cls(client=client, config=config)

    # -- Properties --

    @property
    def last_query_id(self) -> str | None:
        """The query_id from the most recent query, for feedback."""
        return self._last_query_id

    @property
    def config(self) -> QortexMemoryConfig:
        """The current config."""
        return self._config

    # -- Internal helpers --

    async def _extract_last_message(self, model_context: Any) -> str:
        """Extract the text of the last user message from a model context.

        Handles AutoGen's ChatCompletionContext (async get_messages())
        and plain lists of message dicts.
        """
        messages: list[Any] = []

        if hasattr(model_context, "get_messages"):
            messages = await model_context.get_messages()
        elif isinstance(model_context, list):
            messages = model_context
        else:
            return ""

        # Walk backwards to find the last user message
        for msg in reversed(messages):
            # AutoGen message objects
            if hasattr(msg, "content") and hasattr(msg, "source"):
                if msg.source == "user" or type(msg).__name__ == "UserMessage":
                    return str(msg.content)
            # AutoGen LLMMessage types (UserMessage has role)
            elif hasattr(msg, "content"):
                type_name = type(msg).__name__
                if type_name == "UserMessage":
                    return str(msg.content)
            # Plain dicts
            elif isinstance(msg, dict):
                role = msg.get("role", "")
                if role == "user":
                    return str(msg.get("content", ""))

        # Fallback: just use the last message regardless of role
        if messages:
            last = messages[-1]
            if hasattr(last, "content"):
                return str(last.content)
            elif isinstance(last, dict):
                return str(last.get("content", ""))

        return ""
