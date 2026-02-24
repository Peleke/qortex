"""Starlette ASGI application factory for the qortex REST API."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from starlette.applications import Starlette

from qortex.api.middleware import AuthConfig, add_middleware
from qortex.api.routes import build_routes
from qortex.service import QortexService


@asynccontextmanager
async def lifespan(app: Starlette):
    """App lifespan: lazy-init service when not pre-configured, cleanup pool on shutdown."""
    if getattr(app.state, "service", None) is None:
        store_backend = os.environ.get("QORTEX_STORE", "sqlite")
        if store_backend == "postgres":
            app.state.service = await QortexService.async_from_env()
        else:
            app.state.service = QortexService.from_env()
    yield
    # Shutdown order: consumers first, then the pool they depend on.
    intero = getattr(app.state.service, "interoception", None)
    if intero is not None and hasattr(intero, "shutdown"):
        import asyncio

        if asyncio.iscoroutinefunction(intero.shutdown):
            await intero.shutdown()
        else:
            intero.shutdown()

    from qortex.core.pool import close_shared_pool

    await close_shared_pool()


def create_app(
    service: QortexService | None = None,
    auth_config: AuthConfig | None = None,
) -> Starlette:
    """Create the ASGI application.

    Args:
        service: Pre-configured QortexService. If None, created during lifespan.
        auth_config: Auth configuration. If None, loads from env vars.
    """
    routes = build_routes()
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.service = service

    add_middleware(app, auth_config=auth_config)

    return app
