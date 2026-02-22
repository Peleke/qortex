"""HttpQortexClient: async HTTP client for the qortex REST API.

Uses httpx.AsyncClient for native async I/O. No threadpool overhead.
All framework adapters (Agno, AutoGen, LangChain, Mastra, CrewAI) can
use this directly in their async methods.

Install: pip install qortex[http-client]
"""

from __future__ import annotations

from typing import Any

from qortex.client import (
    DomainInfo,
    EdgeItem,
    ExploreResult,
    FeedbackResult,
    IngestResult,
    NodeItem,
    QueryItem,
    QueryResult,
    RuleItem,
    RulesResult,
    StatusResult,
)


class HttpQortexClient:
    """Async HTTP client for the qortex REST API.

    Auth modes (mutually exclusive):
    - ``api_key``: sent as ``Authorization: Bearer <key>``
    - ``hmac_secret``: signs each request body with HMAC-SHA256
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8400",
        api_key: str | None = None,
        hmac_secret: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        import httpx

        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._hmac_secret: bytes | None = (
            hmac_secret.encode() if hmac_secret else None
        )

        event_hooks: dict[str, list] = {}
        if self._hmac_secret:
            event_hooks["request"] = [self._sign_request]

        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            event_hooks=event_hooks,
        )

    async def _sign_request(self, request: Any) -> None:
        """Add HMAC-SHA256 signature headers to outgoing requests."""
        import hashlib
        import hmac
        import time

        if self._hmac_secret is None:
            return

        timestamp = str(int(time.time()))
        body = request.content or b""
        message = f"{timestamp}.".encode() + body
        signature = hmac.new(
            self._hmac_secret, message, hashlib.sha256
        ).hexdigest()

        request.headers["X-Qortex-Timestamp"] = timestamp
        request.headers["X-Qortex-Signature"] = signature

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ------------------------------------------------------------------
    # QortexClient protocol methods (async)
    # ------------------------------------------------------------------

    async def query(
        self,
        context: str,
        domains: list[str] | None = None,
        top_k: int = 20,
        min_confidence: float = 0.0,
    ) -> QueryResult:
        resp = await self._client.post(
            "/v1/query",
            json={
                "context": context,
                "domains": domains,
                "top_k": top_k,
                "min_confidence": min_confidence,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return QueryResult(
            items=[QueryItem(**item) for item in data.get("items", [])],
            query_id=data.get("query_id", ""),
            rules=[RuleItem(**r) for r in data.get("rules", [])],
        )

    async def feedback(
        self,
        query_id: str,
        outcomes: dict[str, str],
        source: str = "unknown",
    ) -> FeedbackResult:
        resp = await self._client.post(
            "/v1/feedback",
            json={
                "query_id": query_id,
                "outcomes": outcomes,
                "source": source,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return FeedbackResult(
            status=data.get("status", ""),
            query_id=data.get("query_id", ""),
            outcome_count=data.get("outcome_count", 0),
            source=data.get("source", ""),
        )

    async def ingest(
        self,
        source_path: str,
        domain: str,
        source_type: str | None = None,
    ) -> IngestResult:
        payload: dict[str, Any] = {
            "source_path": source_path,
            "domain": domain,
        }
        if source_type is not None:
            payload["source_type"] = source_type

        resp = await self._client.post("/v1/ingest", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return IngestResult(
            domain=data.get("domain", ""),
            source=data.get("source", ""),
            concepts=data.get("concepts", 0),
            edges=data.get("edges", 0),
            rules=data.get("rules", 0),
            warnings=data.get("warnings", []),
        )

    async def ingest_text(
        self,
        text: str,
        domain: str,
        format: str = "text",
        name: str | None = None,
    ) -> IngestResult:
        payload: dict[str, Any] = {
            "text": text,
            "domain": domain,
            "format": format,
        }
        if name is not None:
            payload["name"] = name

        resp = await self._client.post("/v1/ingest/text", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return IngestResult(
            domain=data.get("domain", ""),
            source=data.get("source", ""),
            concepts=data.get("concepts", 0),
            edges=data.get("edges", 0),
            rules=data.get("rules", 0),
            warnings=data.get("warnings", []),
        )

    async def ingest_structured(
        self,
        concepts: list[dict[str, Any]],
        domain: str,
        edges: list[dict[str, Any]] | None = None,
        rules: list[dict[str, Any]] | None = None,
    ) -> IngestResult:
        payload: dict[str, Any] = {
            "concepts": concepts,
            "domain": domain,
        }
        if edges is not None:
            payload["edges"] = edges
        if rules is not None:
            payload["rules"] = rules

        resp = await self._client.post("/v1/ingest/structured", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return IngestResult(
            domain=data.get("domain", ""),
            source=data.get("source", ""),
            concepts=data.get("concepts", 0),
            edges=data.get("edges", 0),
            rules=data.get("rules", 0),
            warnings=data.get("warnings", []),
        )

    async def domains(self) -> list[DomainInfo]:
        resp = await self._client.get("/v1/domains")
        resp.raise_for_status()
        data = resp.json()
        return [DomainInfo(**d) for d in data.get("domains", [])]

    async def status(self) -> StatusResult:
        resp = await self._client.get("/v1/status")
        resp.raise_for_status()
        data = resp.json()
        return StatusResult(
            status=data.get("status", ""),
            backend=data.get("backend", ""),
            vector_index=data.get("vector_index"),
            vector_search=data.get("vector_search", False),
            graph_algorithms=data.get("graph_algorithms", False),
            domain_count=data.get("domain_count", 0),
            embedding_model=data.get("embedding_model"),
        )

    async def explore(
        self,
        node_id: str,
        depth: int = 1,
    ) -> ExploreResult | None:
        resp = await self._client.post(
            "/v1/explore",
            json={"node_id": node_id, "depth": depth},
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()

        node_data = data["node"]
        return ExploreResult(
            node=NodeItem(**node_data),
            edges=[EdgeItem(**e) for e in data.get("edges", [])],
            rules=[RuleItem(**r) for r in data.get("rules", [])],
            neighbors=[NodeItem(**n) for n in data.get("neighbors", [])],
        )

    async def rules(
        self,
        domains: list[str] | None = None,
        concept_ids: list[str] | None = None,
        categories: list[str] | None = None,
        include_derived: bool = True,
        min_confidence: float = 0.0,
    ) -> RulesResult:
        resp = await self._client.post(
            "/v1/rules",
            json={
                "domains": domains,
                "concept_ids": concept_ids,
                "categories": categories,
                "include_derived": include_derived,
                "min_confidence": min_confidence,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return RulesResult(
            rules=[RuleItem(**r) for r in data.get("rules", [])],
            domain_count=data.get("domain_count", 0),
            projection=data.get("projection", "rules"),
        )

    async def ingest_database(
        self,
        connection_string: str,
        source_id: str,
        domain_map: dict[str, str] | None = None,
        embed_catalog_tables: bool = True,
        extract_rules: bool = True,
    ) -> dict[str, int]:
        raise NotImplementedError(
            "ingest_database is not supported over HTTP. "
            "Use source_connect + source_sync instead, or call ingest_database "
            "on a LocalQortexClient."
        )
