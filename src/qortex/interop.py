"""Consumer interop protocol for qortex seed distribution.

Implements the hybrid pull/push model for ANY consumer:
- Pull: consumers scan pending/ directory on invocation
- Push: qortex appends to signal log for reactive consumers
- Audit: full event history in append-only log

Any system that understands the format can consume seeds:
- buildlog (primary consumer)
- MCP servers
- Agent frameworks
- CI/CD pipelines
- Custom tooling

Config is read from ~/.claude/qortex-consumers.yaml with sensible defaults.
The default paths use ~/.qortex/ as the namespace (not buildlog-specific).

SECURITY CONSIDERATIONS:
- The pending/ directory is a hot injection point for agents
- Consumers MUST validate seed file contents before processing
- Path traversal attacks: filenames are sanitized, paths are validated
- Signal log is append-only, but consumers should not trust event data blindly
- Consider file permissions on the interop directories (0700 recommended)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# =============================================================================
# Config
# =============================================================================


@dataclass
class SeedsConfig:
    """Paths for seed file directories."""

    pending: Path = field(default_factory=lambda: Path("~/.qortex/seeds/pending"))
    processed: Path = field(default_factory=lambda: Path("~/.qortex/seeds/processed"))
    failed: Path = field(default_factory=lambda: Path("~/.qortex/seeds/failed"))

    def __post_init__(self):
        self.pending = Path(self.pending).expanduser()
        self.processed = Path(self.processed).expanduser()
        self.failed = Path(self.failed).expanduser()

    def ensure_dirs(self) -> None:
        """Create all directories if they don't exist."""
        for d in [self.pending, self.processed, self.failed]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class SignalsConfig:
    """Paths for signal files."""

    projections: Path = field(
        default_factory=lambda: Path("~/.qortex/signals/projections.jsonl")
    )

    def __post_init__(self):
        self.projections = Path(self.projections).expanduser()

    def ensure_dirs(self) -> None:
        """Create parent directories if they don't exist."""
        self.projections.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class InteropConfig:
    """Consumer interop config for qortex seed distribution.

    Any consumer can read seeds from the pending directory and
    subscribe to the signal log for real-time events.
    """

    seeds: SeedsConfig = field(default_factory=SeedsConfig)
    signals: SignalsConfig = field(default_factory=SignalsConfig)

    def ensure_dirs(self) -> None:
        """Create all directories."""
        self.seeds.ensure_dirs()
        self.signals.ensure_dirs()


_CONFIG_PATH = Path("~/.claude/qortex-consumers.yaml").expanduser()


def get_interop_config(config_path: Path | str | None = None) -> InteropConfig:
    """Load consumer interop config.

    Reads from ~/.claude/qortex-consumers.yaml by default.
    Returns defaults if config file doesn't exist.

    Args:
        config_path: Optional override for config file path.
    """
    path = Path(config_path).expanduser() if config_path else _CONFIG_PATH

    if not path.exists():
        return InteropConfig()

    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return InteropConfig()

    seeds_data = data.get("seeds", {})
    signals_data = data.get("signals", {})

    seeds = SeedsConfig(
        pending=seeds_data.get("pending", "~/.qortex/seeds/pending"),
        processed=seeds_data.get("processed", "~/.qortex/seeds/processed"),
        failed=seeds_data.get("failed", "~/.qortex/seeds/failed"),
    )
    signals = SignalsConfig(
        projections=signals_data.get("projections", "~/.qortex/signals/projections.jsonl"),
    )

    return InteropConfig(seeds=seeds, signals=signals)


def write_config(config: InteropConfig, config_path: Path | str | None = None) -> Path:
    """Write config to file.

    Args:
        config: The config to write.
        config_path: Optional override for config file path.

    Returns:
        Path to the written config file.
    """
    path = Path(config_path).expanduser() if config_path else _CONFIG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "seeds": {
            "pending": str(config.seeds.pending),
            "processed": str(config.seeds.processed),
            "failed": str(config.seeds.failed),
        },
        "signals": {
            "projections": str(config.signals.projections),
        },
    }
    path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
    return path


# =============================================================================
# Signal Events
# =============================================================================


@dataclass
class ProjectionEvent:
    """Event emitted when qortex completes a projection.

    Consumers can subscribe to the signal log to react to new projections.
    The event contains all metadata needed to locate and process the seed.
    """

    event: str = "projection_complete"
    persona: str = ""
    domain: str = ""
    path: str = ""
    rule_count: int = 0
    ts: str = ""
    source: str = "qortex"
    source_version: str = "0.1.0"
    # Extensible: consumers can add their own fields
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "event": self.event,
            "persona": self.persona,
            "domain": self.domain,
            "path": self.path,
            "rule_count": self.rule_count,
            "ts": self.ts,
            "source": self.source,
            "source_version": self.source_version,
        }
        if self.extra:
            d.update(self.extra)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectionEvent:
        """Parse an event from a dict."""
        known_keys = {"event", "persona", "domain", "path", "rule_count", "ts", "source", "source_version"}
        extra = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            event=data.get("event", "projection_complete"),
            persona=data.get("persona", ""),
            domain=data.get("domain", ""),
            path=data.get("path", ""),
            rule_count=data.get("rule_count", 0),
            ts=data.get("ts", ""),
            source=data.get("source", "qortex"),
            source_version=data.get("source_version", "0.1.0"),
            extra=extra,
        )


def append_signal(event: ProjectionEvent, config: InteropConfig | None = None) -> None:
    """Append an event to the signal log.

    The signal log is append-only JSONL. Consumers can:
    - Tail the file for real-time events
    - Read from the beginning for full history
    - Use file position to track consumption progress

    Args:
        event: The event to append.
        config: Interop config. If None, loads from default location.
    """
    if config is None:
        config = get_interop_config()

    config.signals.ensure_dirs()

    with config.signals.projections.open("a") as f:
        f.write(json.dumps(event.to_dict()) + "\n")


def read_signals(
    config: InteropConfig | None = None,
    since: datetime | None = None,
    event_types: list[str] | None = None,
) -> list[ProjectionEvent]:
    """Read events from the signal log.

    Args:
        config: Interop config. If None, loads from default location.
        since: Only return events after this timestamp.
        event_types: Only return events of these types.

    Returns:
        List of events, oldest first.
    """
    if config is None:
        config = get_interop_config()

    if not config.signals.projections.exists():
        return []

    events = []
    for line in config.signals.projections.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            event = ProjectionEvent.from_dict(data)

            # Filter by timestamp
            if since and event.ts:
                try:
                    event_ts = datetime.fromisoformat(event.ts.replace("Z", "+00:00"))
                    # Normalize both to aware datetimes for comparison
                    if event_ts.tzinfo is None:
                        event_ts = event_ts.replace(tzinfo=UTC)
                    since_aware = since if since.tzinfo else since.replace(tzinfo=UTC)
                    if event_ts <= since_aware:
                        continue
                except ValueError:
                    pass

            # Filter by event type
            if event_types and event.event not in event_types:
                continue

            events.append(event)
        except json.JSONDecodeError:
            continue
    return events


# =============================================================================
# Seed File Writing
# =============================================================================


def _sanitize_filename(name: str) -> str:
    """Sanitize a string for use in a filename.

    Removes path separators and other dangerous characters.
    """
    # Remove path separators and null bytes
    dangerous = ['/', '\\', '\x00', '..', '~']
    result = name
    for char in dangerous:
        result = result.replace(char, '_')
    # Only allow alphanumeric, underscore, dash, dot
    result = ''.join(c if c.isalnum() or c in '_-.' else '_' for c in result)
    # Collapse multiple underscores
    while '__' in result:
        result = result.replace('__', '_')
    return result.strip('_') or 'unnamed'


def generate_seed_filename(persona: str, timestamp: datetime | None = None) -> str:
    """Generate a seed filename with timestamp.

    Format: {persona}_{YYYY-MM-DDTHH-MM-SS}.yaml
    Colons replaced with dashes for filesystem safety.
    Persona is sanitized to prevent path traversal.

    Args:
        persona: The persona name (will be sanitized).
        timestamp: The timestamp. If None, uses current UTC time.

    Returns:
        Filename like "qortex_impl_hiding_2026-02-05T14-30-00.yaml"
    """
    if timestamp is None:
        timestamp = datetime.now(UTC)

    safe_persona = _sanitize_filename(persona)
    ts_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    return f"{safe_persona}_{ts_str}.yaml"


def _validate_path_in_directory(path: Path, directory: Path) -> bool:
    """Validate that a path is within the expected directory.

    Prevents path traversal attacks.
    """
    try:
        path.resolve().relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def write_seed_to_pending(
    seed_data: dict[str, Any],
    persona: str,
    domain: str,
    config: InteropConfig | None = None,
    emit_signal: bool = True,
    extra_event_data: dict[str, Any] | None = None,
) -> Path:
    """Write a seed file to the pending directory and optionally emit signal.

    This is the main entry point for qortex to publish seeds. Consumers
    can then pick up the seed from pending/ and process it.

    Security: persona is sanitized to prevent path traversal. The generated
    path is validated to be within the pending directory.

    Args:
        seed_data: The seed dict to write (from serialize_ruleset()).
        persona: The persona name (will be sanitized).
        domain: The domain name.
        config: Interop config. If None, loads from default location.
        emit_signal: Whether to append a projection_complete event to signal log.
        extra_event_data: Additional fields to include in the signal event.

    Returns:
        Path to the written seed file.

    Raises:
        ValueError: If the generated path escapes the pending directory.
    """
    if config is None:
        config = get_interop_config()

    config.seeds.ensure_dirs()

    timestamp = datetime.now(UTC)
    filename = generate_seed_filename(persona, timestamp)  # Sanitizes persona
    seed_path = config.seeds.pending / filename

    # Security: validate path is within pending directory
    if not _validate_path_in_directory(seed_path, config.seeds.pending):
        raise ValueError(f"Generated path escapes pending directory: {seed_path}")

    # Write YAML
    seed_path.write_text(yaml.dump(seed_data, default_flow_style=False, sort_keys=False))

    # Emit signal
    if emit_signal:
        event = ProjectionEvent(
            event="projection_complete",
            persona=persona,
            domain=domain,
            path=str(seed_path),
            rule_count=seed_data.get("metadata", {}).get("rule_count", len(seed_data.get("rules", []))),
            ts=timestamp.isoformat(),
            source="qortex",
            source_version=seed_data.get("metadata", {}).get("source_version", "0.1.0"),
            extra=extra_event_data or {},
        )
        append_signal(event, config)

    return seed_path


def list_pending_seeds(config: InteropConfig | None = None) -> list[Path]:
    """List all seed files in the pending directory.

    Consumers can use this to discover seeds awaiting processing.

    Args:
        config: Interop config. If None, loads from default location.

    Returns:
        List of paths to pending seed files, sorted by name (oldest first).
    """
    if config is None:
        config = get_interop_config()

    if not config.seeds.pending.exists():
        return []

    return sorted(config.seeds.pending.glob("*.yaml"))


def list_processed_seeds(config: InteropConfig | None = None) -> list[Path]:
    """List all seed files in the processed directory.

    Args:
        config: Interop config. If None, loads from default location.

    Returns:
        List of paths to processed seed files, sorted by name.
    """
    if config is None:
        config = get_interop_config()

    if not config.seeds.processed.exists():
        return []

    return sorted(config.seeds.processed.glob("*.yaml"))


def list_failed_seeds(config: InteropConfig | None = None) -> list[tuple[Path, str | None]]:
    """List all seed files in the failed directory with error messages.

    Args:
        config: Interop config. If None, loads from default location.

    Returns:
        List of (seed_path, error_message) tuples. Error is None if no .error sidecar.
    """
    if config is None:
        config = get_interop_config()

    if not config.seeds.failed.exists():
        return []

    results = []
    for seed_path in sorted(config.seeds.failed.glob("*.yaml")):
        error_path = seed_path.with_suffix(".error")
        error_msg = error_path.read_text() if error_path.exists() else None
        results.append((seed_path, error_msg))
    return results
