"""Route handlers for the qortex REST API.

Thin layer: parse request, call QortexService, return JSON.
All service methods are async — route handlers await them directly.
"""

from __future__ import annotations

from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from qortex.service import QortexService


def _get_service(request: Request) -> QortexService:
    return request.app.state.service


def _error(message: str, status: int = 400) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


# ------------------------------------------------------------------
# Health / Status / Domains
# ------------------------------------------------------------------


async def health_handler(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


async def status_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    result = await service.status()
    return JSONResponse(result)


async def domains_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    result = await service.domains()
    return JSONResponse(result)


async def stats_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    result = await service.stats()
    return JSONResponse(result)


# ------------------------------------------------------------------
# Query / Feedback
# ------------------------------------------------------------------


async def query_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    context = body.get("context")
    if not context:
        return _error("'context' is required")

    result = await service.query(
        context=context,
        domains=body.get("domains"),
        top_k=body.get("top_k", 20),
        min_confidence=body.get("min_confidence", 0.0),
        mode=body.get("mode", "auto"),
    )
    return JSONResponse(result)


async def feedback_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    query_id = body.get("query_id")
    outcomes = body.get("outcomes")
    if not query_id or not outcomes:
        return _error("'query_id' and 'outcomes' are required")

    result = await service.feedback(
        query_id=query_id,
        outcomes=outcomes,
        source=body.get("source", "http"),
    )
    return JSONResponse(result)


# ------------------------------------------------------------------
# Ingest
# ------------------------------------------------------------------


async def ingest_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    source_path = body.get("source_path")
    domain = body.get("domain")
    if not source_path or not domain:
        return _error("'source_path' and 'domain' are required")

    result = await service.ingest(
        source_path=source_path,
        domain=domain,
        source_type=body.get("source_type"),
    )
    return JSONResponse(result)


async def ingest_text_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    text = body.get("text")
    domain = body.get("domain")
    if not text or not domain:
        return _error("'text' and 'domain' are required")

    result = await service.ingest_text(
        text=text,
        domain=domain,
        format=body.get("format", "text"),
        name=body.get("name"),
    )
    return JSONResponse(result)


async def ingest_structured_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    concepts = body.get("concepts")
    domain = body.get("domain")
    if not concepts or not domain:
        return _error("'concepts' and 'domain' are required")

    result = await service.ingest_structured(
        concepts=concepts,
        domain=domain,
        edges=body.get("edges"),
        rules=body.get("rules"),
    )
    return JSONResponse(result)


async def ingest_message_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    text = body.get("text")
    session_id = body.get("session_id")
    if not text or not session_id:
        return _error("'text' and 'session_id' are required")

    result = await service.ingest_message(
        text=text,
        session_id=session_id,
        role=body.get("role", "user"),
        domain=body.get("domain", "session"),
    )
    return JSONResponse(result)


# ------------------------------------------------------------------
# Explore / Rules / Compare
# ------------------------------------------------------------------


async def explore_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    node_id = body.get("node_id")
    if not node_id:
        return _error("'node_id' is required")

    result = await service.explore(
        node_id=node_id,
        depth=body.get("depth", 1),
    )
    if result is None:
        return JSONResponse({"error": f"Node '{node_id}' not found"}, status_code=404)
    return JSONResponse(result)


async def rules_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    result = await service.rules(
        domains=body.get("domains"),
        concept_ids=body.get("concept_ids"),
        categories=body.get("categories"),
        include_derived=body.get("include_derived", True),
        min_confidence=body.get("min_confidence", 0.0),
    )
    return JSONResponse(result)


# ------------------------------------------------------------------
# Learning
# ------------------------------------------------------------------


async def learning_select_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    learner = body.get("learner")
    candidates = body.get("candidates")
    if not learner or not candidates:
        return _error("'learner' and 'candidates' are required")

    result = await service.learning_select(
        learner=learner,
        candidates=candidates,
        context=body.get("context"),
        k=body.get("k", 1),
        token_budget=body.get("token_budget", 0),
        min_pulls=body.get("min_pulls", 0),
        seed_arms=body.get("seed_arms"),
        seed_boost=body.get("seed_boost"),
    )
    return JSONResponse(result)


async def learning_observe_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    learner = body.get("learner")
    arm_id = body.get("arm_id")
    if not learner or not arm_id:
        return _error("'learner' and 'arm_id' are required")

    result = await service.learning_observe(
        learner=learner,
        arm_id=arm_id,
        outcome=body.get("outcome", ""),
        reward=body.get("reward", 0.0),
        context=body.get("context"),
    )
    return JSONResponse(result)


async def learning_posteriors_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    learner = request.path_params["learner"]
    result = await service.learning_posteriors(learner=learner)
    return JSONResponse(result)


async def learning_metrics_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    learner = request.path_params["learner"]
    window = request.query_params.get("window")
    result = await service.learning_metrics(
        learner=learner,
        window=int(window) if window else None,
    )
    return JSONResponse(result)


async def learning_session_start_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    learner = body.get("learner")
    session_name = body.get("session_name")
    if not learner or not session_name:
        return _error("'learner' and 'session_name' are required")

    result = await service.learning_session_start(
        learner=learner,
        session_name=session_name,
    )
    return JSONResponse(result)


async def learning_session_end_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    session_id = body.get("session_id")
    if not session_id:
        return _error("'session_id' is required")

    result = await service.learning_session_end(session_id=session_id)
    return JSONResponse(result)


async def learning_reset_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    learner = body.get("learner")
    if not learner:
        return _error("'learner' is required")

    result = await service.learning_reset(
        learner=learner,
        arm_ids=body.get("arm_ids"),
        context=body.get("context"),
    )
    return JSONResponse(result)


# ------------------------------------------------------------------
# Sources
# ------------------------------------------------------------------


async def source_connect_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    source_id = body.get("source_id")
    connection_string = body.get("connection_string")
    if not source_id or not connection_string:
        return _error("'source_id' and 'connection_string' are required")

    result = await service.source_connect(
        source_id=source_id,
        connection_string=connection_string,
        schemas=body.get("schemas"),
        domain_map=body.get("domain_map"),
    )
    return JSONResponse(result)


async def source_sync_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    source_id = request.path_params["source_id"]
    try:
        body = await request.json()
    except Exception:
        body = {}

    result = await service.source_sync(
        source_id=source_id,
        tables=body.get("tables"),
        mode=body.get("mode", "full"),
    )
    return JSONResponse(result)


async def source_list_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    result = await service.source_list()
    return JSONResponse(result)


async def source_disconnect_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    source_id = request.path_params["source_id"]
    result = await service.source_disconnect(source_id)
    return JSONResponse(result)


# ------------------------------------------------------------------
# Admin / Migration
# ------------------------------------------------------------------


async def migrate_vec_handler(request: Request) -> JSONResponse:
    service = _get_service(request)
    try:
        body = await request.json()
    except Exception:
        return _error("Invalid JSON body")

    source = body.get("source")
    if not source:
        return _error("'source' is required (sqlite, pgvector, numpy)")

    result = await service.migrate_vec(
        source_type=source,
        batch_size=body.get("batch_size", 500),
        dry_run=body.get("dry_run", False),
    )
    return JSONResponse(result)


# ------------------------------------------------------------------
# Route table
# ------------------------------------------------------------------


def build_routes() -> list[Route]:
    """Build the list of routes for the Starlette app."""
    return [
        # Health / Status
        Route("/v1/health", health_handler, methods=["GET"]),
        Route("/v1/status", status_handler, methods=["GET"]),
        Route("/v1/domains", domains_handler, methods=["GET"]),
        Route("/v1/stats", stats_handler, methods=["GET"]),
        # Query / Feedback
        Route("/v1/query", query_handler, methods=["POST"]),
        Route("/v1/feedback", feedback_handler, methods=["POST"]),
        # Ingest
        Route("/v1/ingest", ingest_handler, methods=["POST"]),
        Route("/v1/ingest/text", ingest_text_handler, methods=["POST"]),
        Route("/v1/ingest/structured", ingest_structured_handler, methods=["POST"]),
        Route("/v1/ingest/message", ingest_message_handler, methods=["POST"]),
        # Explore / Rules
        Route("/v1/explore", explore_handler, methods=["POST"]),
        Route("/v1/rules", rules_handler, methods=["POST"]),
        # Learning
        Route("/v1/learning/select", learning_select_handler, methods=["POST"]),
        Route("/v1/learning/observe", learning_observe_handler, methods=["POST"]),
        Route(
            "/v1/learning/{learner}/posteriors",
            learning_posteriors_handler,
            methods=["GET"],
        ),
        Route(
            "/v1/learning/{learner}/metrics",
            learning_metrics_handler,
            methods=["GET"],
        ),
        # Learning sessions / reset
        Route("/v1/learning/sessions/start", learning_session_start_handler, methods=["POST"]),
        Route("/v1/learning/sessions/end", learning_session_end_handler, methods=["POST"]),
        Route("/v1/learning/reset", learning_reset_handler, methods=["POST"]),
        # Sources
        Route("/v1/sources/connect", source_connect_handler, methods=["POST"]),
        Route(
            "/v1/sources/{source_id}/sync",
            source_sync_handler,
            methods=["POST"],
        ),
        Route("/v1/sources", source_list_handler, methods=["GET"]),
        Route(
            "/v1/sources/{source_id}",
            source_disconnect_handler,
            methods=["DELETE"],
        ),
        # Admin / Migration
        Route("/v1/admin/migrate-vec", migrate_vec_handler, methods=["POST"]),
    ]
