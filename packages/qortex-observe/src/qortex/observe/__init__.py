"""qortex observability: event-driven architecture for metrics, traces, and logs.

Public API:
    emit(event)                  Fire-and-forget event emission (no-op if not configured)
    configure(cfg)               Initialize emitter + subscribers (call once at startup)
    reset()                      Reset for testing

Logging (swappable formatter x destination):
    get_logger(name)             Get a structured logger
    register_formatter(n, cls)   Register custom LogFormatter
    register_destination(n, cls) Register custom LogDestination

Modules import `emit` and fire typed events. They don't know about
metrics, traces, or logs. Subscribers handle routing.
"""

from qortex.observe.emitter import configure, emit, is_configured, reset
from qortex.observe.events import (
    BenchLearningCurveRecorded,
    BenchRoundCompleted,
    BufferFlushed,
    CarbonTracked,
    ConceptsExtracted,
    CreditPropagated,
    EdgePromoted,
    EnrichmentCompleted,
    EnrichmentFallback,
    ExtractionPipelineCompleted,
    FactorDriftSnapshot,
    FactorsLoaded,
    FactorsPersisted,
    FactorUpdated,
    FeedbackReceived,
    GraphEdgesCreated,
    GraphNodesCreated,
    InteroceptionShutdown,
    InteroceptionStarted,
    KGCoverageComputed,
    LearningObservationRecorded,
    LearningPosteriorUpdated,
    LearningSelectionMade,
    ManifestIngested,
    MessageIngested,
    OnlineEdgeRecorded,
    OnlineEdgesGenerated,
    PPRConverged,
    PPRDiverged,
    PPRStarted,
    QueryCompleted,
    QueryFailed,
    QueryStarted,
    ToolResultIngested,
    VecIndexUpdated,
    VecSearchCompleted,
    VecSearchResults,
    VecSeedYield,
)
from qortex.observe.logging import (
    LogDestination,
    LogFormatter,
    get_logger,
    register_destination,
    register_formatter,
)
from qortex.observe.metrics_schema import METRICS, MetricDef, MetricType
from qortex.observe.snapshot import config_snapshot_hash
from qortex.observe.tracing import OverheadTimer, SelectiveSpanProcessor, traced

__all__ = [
    # Core API
    "emit",
    "configure",
    "is_configured",
    "reset",
    # Logging (swappable)
    "get_logger",
    "LogFormatter",
    "LogDestination",
    "register_formatter",
    "register_destination",
    # Query lifecycle
    "QueryStarted",
    "QueryCompleted",
    "QueryFailed",
    # PPR convergence
    "PPRStarted",
    "PPRConverged",
    "PPRDiverged",
    # Teleportation factors
    "FactorUpdated",
    "FactorsPersisted",
    "FactorsLoaded",
    "FactorDriftSnapshot",
    # Edge promotion
    "OnlineEdgeRecorded",
    "EdgePromoted",
    "BufferFlushed",
    # Retrieval
    "VecSearchCompleted",
    "OnlineEdgesGenerated",
    "KGCoverageComputed",
    "FeedbackReceived",
    # Lifecycle
    "InteroceptionStarted",
    "InteroceptionShutdown",
    # Enrichment
    "EnrichmentCompleted",
    "EnrichmentFallback",
    # Vector index
    "VecIndexUpdated",
    "VecSearchResults",
    "VecSeedYield",
    # Online Indexing
    "MessageIngested",
    "ToolResultIngested",
    # Concept Extraction
    "ConceptsExtracted",
    "ExtractionPipelineCompleted",
    # Ingestion
    "ManifestIngested",
    "GraphNodesCreated",
    "GraphEdgesCreated",
    # Carbon
    "CarbonTracked",
    # Learning
    "LearningSelectionMade",
    "LearningObservationRecorded",
    "LearningPosteriorUpdated",
    # Credit Propagation
    "CreditPropagated",
    # Benchmarks
    "BenchRoundCompleted",
    "BenchLearningCurveRecorded",
    # Metric schema
    "MetricDef",
    "MetricType",
    "METRICS",
    # Tracing
    "traced",
    "OverheadTimer",
    "SelectiveSpanProcessor",
    # Config snapshot
    "config_snapshot_hash",
]
