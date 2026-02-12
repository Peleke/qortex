"""JSONL file subscriber: writes every event to a JSONL file.

Loki-ready format. Each line is a complete JSON object with
event type name and all fields.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from qortex.observability.events import (
    BufferFlushed,
    EdgePromoted,
    EnrichmentCompleted,
    EnrichmentFallback,
    FactorDriftSnapshot,
    FactorsLoaded,
    FactorsPersisted,
    FactorUpdated,
    FeedbackReceived,
    InteroceptionShutdown,
    InteroceptionStarted,
    KGCoverageComputed,
    LearningObservationRecorded,
    LearningPosteriorUpdated,
    LearningSelectionMade,
    ManifestIngested,
    OnlineEdgeRecorded,
    OnlineEdgesGenerated,
    PPRConverged,
    PPRDiverged,
    PPRStarted,
    QueryCompleted,
    QueryFailed,
    QueryStarted,
    VecSearchCompleted,
)
from qortex.observability.linker import QortexEventLinker
from qortex.observability.sinks.jsonl_sink import JsonlSink

# All event types that should be written to JSONL
_ALL_EVENTS = (
    QueryStarted,
    QueryCompleted,
    QueryFailed,
    PPRStarted,
    PPRConverged,
    PPRDiverged,
    FactorUpdated,
    FactorsPersisted,
    FactorsLoaded,
    FactorDriftSnapshot,
    OnlineEdgeRecorded,
    EdgePromoted,
    BufferFlushed,
    VecSearchCompleted,
    OnlineEdgesGenerated,
    KGCoverageComputed,
    FeedbackReceived,
    InteroceptionStarted,
    InteroceptionShutdown,
    EnrichmentCompleted,
    EnrichmentFallback,
    ManifestIngested,
    LearningSelectionMade,
    LearningObservationRecorded,
    LearningPosteriorUpdated,
)


def register_jsonl_subscriber(path: str) -> None:
    """Register a catch-all subscriber that writes every event to JSONL."""
    sink = JsonlSink(Path(path))

    @QortexEventLinker.on(*_ALL_EVENTS)
    def _write_jsonl(event: object) -> None:
        sink.write({"event": type(event).__name__, **asdict(event)})  # type: ignore[arg-type]
