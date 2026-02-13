"""Interoception layer: the control surface between retrieval and feedback.

In neuroscience, interoception is the sense of internal body state. The amygdala
uses it to tag memories with valence that biases future retrieval. This module
does exactly the same: it provides behavioral bias (teleportation factors) to the
retrieval pipeline without the pipeline knowing where the bias comes from.

Two-protocol design:

    [OutcomeSource]  ──report()──→  [InteroceptionProvider]  ──get_seed_weights()──→  [GraphRAGAdapter]
      openclaw arms                   factor storage                                    PPR pipeline
      buildlog rewards                edge buffer
      cadence events                  lifecycle mgmt
      MCP tool calls

InteroceptionProvider: "how to bias retrieval" (consumed by GraphRAGAdapter).
OutcomeSource: "where outcomes come from" (feeds into InteroceptionProvider).

Today OutcomeSource is the MCP tool interface (qortex_feedback). But the protocol
means any system can inject outcomes: openclaw's learning package, buildlog rewards,
cadence's ambient events, or a future network-backed federation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from qortex.observability import emit
from qortex.observability.events import InteroceptionShutdown, InteroceptionStarted
from qortex.observability.logging import get_logger

if TYPE_CHECKING:
    from qortex.hippocampus.buffer import EdgePromotionBuffer
    from qortex.hippocampus.factors import TeleportationFactors

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Outcome Source (where feedback comes from)
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    """A single outcome observation from any consumer."""

    query_id: str
    item_id: str
    result: str  # "accepted" | "rejected" | "partial"
    source: str  # "openclaw" | "buildlog" | "manual" | "cadence"
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class OutcomeSource(Protocol):
    """Where outcomes come from. Abstract so the source can be swapped.

    Today: MCP tool call (qortex_feedback).
    Tomorrow: openclaw learning package, buildlog rewards, cadence events.
    """

    def report(self, outcomes: list[Outcome]) -> None:
        """Push outcomes into the interoception layer."""
        ...


# ---------------------------------------------------------------------------
# Interoception Provider (how retrieval gets biased)
# ---------------------------------------------------------------------------


@runtime_checkable
class InteroceptionProvider(Protocol):
    """Provides behavioral bias for graph retrieval.

    Consumed by GraphRAGAdapter for:
    - Seed weight computation (PPR personalization vector)
    - Outcome reporting (factor updates from feedback)
    - Online edge recording (buffer for future promotion)
    - Lifecycle management (startup/shutdown with persistence)
    """

    def get_seed_weights(self, seed_ids: list[str]) -> dict[str, float]:
        """Normalized teleportation weights for PPR seeds."""
        ...

    def report_outcome(self, query_id: str, outcomes: dict[str, str]) -> None:
        """Update internal state from feedback.

        Args:
            query_id: The query that produced these results.
            outcomes: Mapping of item_id to "accepted" | "rejected" | "partial".
        """
        ...

    def record_online_edge(self, source_id: str, target_id: str, score: float) -> None:
        """Record an online edge observation for the buffer."""
        ...

    def startup(self) -> None:
        """Load persisted state from disk."""
        ...

    def shutdown(self) -> None:
        """Persist state to disk."""
        ...

    def summary(self) -> dict[str, Any]:
        """Summary stats for monitoring/logging."""
        ...

    @property
    def factors(self) -> TeleportationFactors:
        """Current teleportation factors."""
        ...

    @property
    def buffer(self) -> EdgePromotionBuffer:
        """Current edge promotion buffer."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _teleportation_enabled_default() -> bool:
    """Read teleportation flag from centralized FeatureFlags."""
    from qortex.flags import get_flags

    return get_flags().teleportation


@dataclass
class InteroceptionConfig:
    """Configuration for LocalInteroceptionProvider."""

    factors_path: Path | None = None
    buffer_path: Path | None = None
    auto_flush_threshold: int = 50
    persist_on_update: bool = True
    teleportation_enabled: bool = field(default_factory=_teleportation_enabled_default)


# ---------------------------------------------------------------------------
# Local implementation (in-process, wraps existing components)
# ---------------------------------------------------------------------------


class LocalInteroceptionProvider:
    """Default InteroceptionProvider: in-process, wraps TeleportationFactors + EdgePromotionBuffer.

    Lifecycle:
        startup()  → Load factors + buffer from disk (if paths configured)
        ...queries + feedback...
        shutdown() → Persist both to disk, log summary
    """

    def __init__(self, config: InteroceptionConfig | None = None) -> None:
        from qortex.hippocampus.buffer import EdgePromotionBuffer
        from qortex.hippocampus.factors import TeleportationFactors

        self._config = config or InteroceptionConfig()
        self._factors = TeleportationFactors()
        self._buffer = EdgePromotionBuffer()
        self._started = False
        self._backend: Any = None  # Set by adapter for auto-flush

    def set_backend(self, backend: Any) -> None:
        """Attach a graph backend for edge promotion auto-flush."""
        self._backend = backend

    def startup(self) -> None:
        """Load persisted state from disk (if paths configured)."""
        from qortex.hippocampus.buffer import EdgePromotionBuffer
        from qortex.hippocampus.factors import TeleportationFactors

        if self._config.factors_path is not None:
            self._factors = TeleportationFactors.load(self._config.factors_path)
            logger.info(
                "interoception.factors.loaded",
                count=len(self._factors.factors),
                path=str(self._config.factors_path),
            )
        else:
            self._factors = TeleportationFactors()

        if self._config.buffer_path is not None:
            self._buffer = EdgePromotionBuffer.load(self._config.buffer_path)
            logger.info(
                "interoception.buffer.loaded",
                count=self._buffer.summary()["buffered_edges"],
                path=str(self._config.buffer_path),
            )
        else:
            self._buffer = EdgePromotionBuffer()

        self._started = True

        emit(InteroceptionStarted(
            factors_loaded=len(self._factors.factors),
            buffer_loaded=self._buffer.summary()["buffered_edges"],
            teleportation_enabled=self._config.teleportation_enabled,
        ))

    def shutdown(self) -> None:
        """Persist state to disk and log summary."""
        if self._config.factors_path is not None:
            self._factors.persist(self._config.factors_path)
            logger.info("interoception.factors.persisted", path=str(self._config.factors_path))

        if self._config.buffer_path is not None:
            self._buffer.persist(self._config.buffer_path)
            logger.info("interoception.buffer.persisted", path=str(self._config.buffer_path))

        s = self.summary()
        logger.info("interoception.shutdown", **s)

        emit(InteroceptionShutdown(
            factors_persisted=len(self._factors.factors),
            buffer_persisted=self._buffer.summary()["buffered_edges"],
            summary=s,
        ))

    def get_seed_weights(self, seed_ids: list[str]) -> dict[str, float]:
        """Normalized teleportation weights for PPR seeds.

        When teleportation is disabled (default), returns uniform weights.
        Factors still accumulate via report_outcome() so enabling the flag
        later uses all historical data.
        """
        if not self._config.teleportation_enabled:
            if not seed_ids:
                return {}
            uniform = 1.0 / len(seed_ids)
            return {nid: uniform for nid in seed_ids}
        return self._factors.weight_seeds(seed_ids)

    def report_outcome(self, query_id: str, outcomes: dict[str, str]) -> None:
        """Update factors from feedback outcomes and optionally persist."""
        updates = self._factors.update(query_id, outcomes)
        if updates and self._config.persist_on_update and self._config.factors_path is not None:
            self._factors.persist(self._config.factors_path)

    def record_online_edge(self, source_id: str, target_id: str, score: float) -> None:
        """Record an online edge observation in the buffer."""
        self._buffer.record(source_id, target_id, score)

        buffered = self._buffer.summary()["buffered_edges"]
        if self._config.auto_flush_threshold > 0 and buffered >= self._config.auto_flush_threshold:
            logger.info(
                "interoception.buffer.threshold_reached",
                buffered_edges=buffered,
                threshold=self._config.auto_flush_threshold,
            )
            # Auto-flush: promote qualifying edges to persistent KG
            if self._backend is not None:
                result = self.flush_buffer(self._backend)
                logger.info(
                    "interoception.buffer.auto_flushed",
                    promoted=result.get("promoted", 0) if isinstance(result, dict) else 0,
                )
            else:
                logger.debug("interoception.buffer.auto_flush_skipped", reason="no_backend")

    def flush_buffer(self, backend: Any, **kwargs: Any) -> Any:
        """Flush the edge buffer, promoting qualifying edges to persistent KG.

        Args:
            backend: GraphBackend to add promoted edges to.
            **kwargs: Passed to EdgePromotionBuffer.flush() (min_hits, min_avg_score).

        Returns:
            PromotionResult from the buffer flush.
        """
        result = self._buffer.flush(backend, **kwargs)

        if self._config.buffer_path is not None:
            self._buffer.persist(self._config.buffer_path)

        return result

    @property
    def factors(self) -> Any:
        """Direct access to TeleportationFactors (for hooks/testing)."""
        return self._factors

    @property
    def buffer(self) -> Any:
        """Direct access to EdgePromotionBuffer (for hooks/testing)."""
        return self._buffer

    def summary(self) -> dict[str, Any]:
        """Merged summary from factors + buffer."""
        return {
            "factors": self._factors.summary(),
            "buffer": self._buffer.summary(),
            "started": self._started,
            "persist_on_update": self._config.persist_on_update,
            "teleportation_enabled": self._config.teleportation_enabled,
        }


# ---------------------------------------------------------------------------
# MCP Outcome Source (thin adapter for MCP tool → interoception)
# ---------------------------------------------------------------------------


class McpOutcomeSource:
    """Default OutcomeSource: bridges MCP tool calls to interoception.

    This is a thin adapter — the MCP tool already does the work.
    Exists so that non-MCP outcome sources (openclaw learning package,
    buildlog rewards, cadence events) can implement the same protocol.
    """

    def __init__(self, interoception: InteroceptionProvider) -> None:
        self._interoception = interoception

    def report(self, outcomes: list[Outcome]) -> None:
        """Group outcomes by query_id and route to interoception."""
        grouped: dict[str, dict[str, str]] = {}
        for outcome in outcomes:
            if outcome.query_id not in grouped:
                grouped[outcome.query_id] = {}
            grouped[outcome.query_id][outcome.item_id] = outcome.result

        for query_id, outcome_map in grouped.items():
            self._interoception.report_outcome(query_id, outcome_map)
