"""CLI configuration via environment variables."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field


def _int_env(var: str, default: int) -> int:
    """Parse an integer from an environment variable with a helpful error on bad input."""
    raw = os.environ.get(var)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as err:
        print(f"Error: {var}={raw!r} is not a valid integer", file=sys.stderr)
        raise SystemExit(1) from err


@dataclass
class MemgraphCredentials:
    """Secure wrapper for Memgraph authentication.

    Password is hidden from repr/str to avoid accidental logging.
    """

    user: str = field(default_factory=lambda: os.environ.get("QORTEX_MEMGRAPH_USER", "qortex"))
    _password: str = field(
        default_factory=lambda: os.environ.get("QORTEX_MEMGRAPH_PASSWORD", "qortex"),
        repr=False,
    )

    @property
    def password(self) -> str:
        """Access password (use sparingly, prefer auth_tuple)."""
        return self._password

    @property
    def auth_tuple(self) -> tuple[str, str]:
        """Get (user, password) tuple for neo4j driver."""
        return (self.user, self._password)

    def __str__(self) -> str:
        return f"MemgraphCredentials(user={self.user!r})"


@dataclass
class QortexConfig:
    """Configuration for the qortex CLI.

    Reads from environment variables with QORTEX_ prefix.
    Falls back to sensible defaults for local development.

    Connection priority:
        1. QORTEX_MEMGRAPH_URI (full bolt:// URI, overrides host/port)
        2. QORTEX_MEMGRAPH_HOST + QORTEX_MEMGRAPH_PORT
    """

    # Connection - URI takes precedence over host/port
    memgraph_uri: str | None = field(default_factory=lambda: os.environ.get("QORTEX_MEMGRAPH_URI"))
    memgraph_host: str = field(
        default_factory=lambda: os.environ.get("QORTEX_MEMGRAPH_HOST", "localhost")
    )
    memgraph_port: int = field(default_factory=lambda: _int_env("QORTEX_MEMGRAPH_PORT", 7687))

    # Credentials (wrapped for security)
    memgraph_credentials: MemgraphCredentials = field(default_factory=MemgraphCredentials)

    # Lab UI
    lab_port: int = field(default_factory=lambda: _int_env("QORTEX_LAB_PORT", 3000))

    # Docker
    compose_file: str = field(
        default_factory=lambda: os.environ.get("QORTEX_COMPOSE_FILE", "docker/docker-compose.yml")
    )

    # API keys (not wrapped since they're optional and used differently)
    anthropic_api_key: str | None = field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )

    def get_memgraph_uri(self) -> str:
        """Get the Memgraph connection URI.

        Returns QORTEX_MEMGRAPH_URI if set, otherwise constructs from host/port.
        """
        if self.memgraph_uri:
            return self.memgraph_uri
        return f"bolt://{self.memgraph_host}:{self.memgraph_port}"


def get_config() -> QortexConfig:
    """Get the current configuration."""
    return QortexConfig()
