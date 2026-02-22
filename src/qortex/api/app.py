"""Starlette ASGI application factory for the qortex REST API."""

from __future__ import annotations

from contextlib import asynccontextmanager

from starlette.applications import Starlette

from qortex.api.middleware import AuthConfig, add_middleware
from qortex.api.routes import build_routes
from qortex.service import QortexService


@asynccontextmanager
async def lifespan(app: Starlette):
    yield


def create_app(
    service: QortexService | None = None,
    auth_config: AuthConfig | None = None,
) -> Starlette:
    """Create the ASGI application.

    Args:
        service: Pre-configured QortexService. If None, creates from env vars.
        auth_config: Auth configuration. If None, loads from env vars.
    """
    if service is None:
        service = QortexService.from_env()

    routes = build_routes()
    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.service = service

    add_middleware(app, auth_config=auth_config)

    return app
