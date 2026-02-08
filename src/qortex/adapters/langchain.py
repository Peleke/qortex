"""LangChain adapter: drop-in BaseRetriever backed by qortex.

Usage:
    from qortex.client import LocalQortexClient
    from qortex.adapters.langchain import QortexRetriever

    client = LocalQortexClient(vector_index, backend, embedding_model)
    retriever = QortexRetriever(client=client, domains=["security"])

    # Works anywhere langchain expects a retriever
    docs = retriever.invoke("What is OAuth2?")
    # → [Document(page_content="...", metadata={score, domain, ...}, id="...")]

Requires: pip install langchain-core
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


def _check_langchain():
    if not _HAS_LANGCHAIN:
        raise ImportError("langchain-core required for QortexRetriever: pip install langchain-core")


if _HAS_LANGCHAIN:

    class QortexRetriever(BaseRetriever):
        """Drop-in langchain BaseRetriever backed by qortex.

        Converts qortex QueryItems to langchain Documents.
        Score is included in metadata (langchain convention).
        """

        client: Any  # QortexClient — Any for Pydantic compatibility
        domains: list[str] | None = None
        top_k: int = 20
        min_confidence: float = 0.0
        feedback_source: str = "langchain"

        # Internal state (not serialized)
        _last_query_id: str | None = None

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun | None = None,
        ) -> list[Document]:
            result = self.client.query(
                context=query,
                domains=self.domains,
                top_k=self.top_k,
                min_confidence=self.min_confidence,
            )
            self._last_query_id = result.query_id

            return [item.to_langchain_document() for item in result.items]

        def feedback(self, outcomes: dict[str, str]) -> None:
            """Report feedback for the last query. Closes the learning loop."""
            if self._last_query_id is None:
                return
            self.client.feedback(
                query_id=self._last_query_id,
                outcomes=outcomes,
                source=self.feedback_source,
            )

else:

    class QortexRetriever:  # type: ignore[no-redef]
        """Stub: langchain-core not installed."""

        def __init__(self, **kwargs):
            _check_langchain()
