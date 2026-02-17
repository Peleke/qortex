"""Edge Promotion Buffer: online-gen edges → persistent KG.

Online-generated cosine-sim edges are ephemeral by default. But edges
that appear frequently across queries represent real relationships that
should be promoted to the persistent KG.

Lifecycle:
    1. Query-time: GraphRAGAdapter records online edges with hit counts + scores
    2. Idle-time: flush() promotes edges exceeding thresholds to persistent KG
    3. Promoted edges appear in the persistent KG for future queries
    4. Buffer persists across process restarts via JSONL file

Research signal: promotion rate = knowledge crystallization rate. As the KG
matures, fewer new edges qualify → system is converging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from qortex.observe import emit
from qortex.observe.events import BufferFlushed, EdgePromoted, OnlineEdgeRecorded
from qortex.observe.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EdgeStats:
    """Stats for a buffered edge observation."""

    hit_count: int = 0
    scores: list[float] = field(default_factory=list)
    last_seen: str = ""  # ISO timestamp

    @property
    def avg_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)


@dataclass
class PromotionResult:
    """Result of a flush() operation."""

    promoted: int
    remaining: int
    details: list[dict[str, Any]] = field(default_factory=list)


class EdgePromotionBuffer:
    """Buffers online-gen edges for batch promotion to persistent KG.

    Edges are keyed by (min_id, max_id) to deduplicate regardless of
    direction — online cosine-sim edges are undirected.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._buffer: dict[tuple[str, str], EdgeStats] = {}
        self._persistence_path = path
        self._total_promoted = 0  # Lifetime counter

        # Lifecycle hooks
        self._hooks: dict[str, list[Any]] = {
            "on_record": [],  # Called when an edge observation is recorded
            "on_promote": [],  # Called for each promoted edge
            "on_flush": [],  # Called after flush completes with PromotionResult
        }

    def record(self, source_id: str, target_id: str, score: float) -> None:
        """Record an online-gen edge observation."""
        key = (min(source_id, target_id), max(source_id, target_id))
        if key not in self._buffer:
            self._buffer[key] = EdgeStats()

        stats = self._buffer[key]
        stats.hit_count += 1
        stats.scores.append(score)
        stats.last_seen = datetime.now(UTC).isoformat()

        emit(
            OnlineEdgeRecorded(
                source_id=source_id,
                target_id=target_id,
                score=score,
                hit_count=stats.hit_count,
                buffer_size=len(self._buffer),
            )
        )

        for hook in self._hooks.get("on_record", []):
            try:
                hook(source_id, target_id, score, stats.hit_count)
            except Exception:
                logger.debug("buffer.hook.failed", hook_event="on_record")

    def get_edge_stats(self, source_id: str, target_id: str) -> EdgeStats | None:
        """Return stats for a buffered edge, or None if not recorded."""
        key = (min(source_id, target_id), max(source_id, target_id))
        return self._buffer.get(key)

    def flush(
        self,
        backend: Any,
        min_hits: int = 3,
        min_avg_score: float = 0.75,
    ) -> PromotionResult:
        """Promote qualifying edges to persistent KG.

        Args:
            backend: GraphBackend to add promoted edges to.
            min_hits: Minimum observation count to qualify.
            min_avg_score: Minimum average cosine similarity.

        Returns:
            PromotionResult with counts and details.
        """
        from qortex.core.models import ConceptEdge, RelationType

        promoted = 0
        details: list[dict[str, Any]] = []
        to_remove: list[tuple[str, str]] = []

        for (src, tgt), stats in self._buffer.items():
            if stats.hit_count >= min_hits and stats.avg_score >= min_avg_score:
                edge = ConceptEdge(
                    source_id=src,
                    target_id=tgt,
                    relation_type=RelationType.SIMILAR_TO,
                    confidence=stats.avg_score,
                    properties={
                        "promoted_from": "online_gen",
                        "hit_count": stats.hit_count,
                        "promoted_at": datetime.now(UTC).isoformat(),
                    },
                )
                try:
                    backend.add_edge(edge)
                    to_remove.append((src, tgt))
                    promoted += 1
                    self._total_promoted += 1

                    detail = {
                        "source": src,
                        "target": tgt,
                        "hits": stats.hit_count,
                        "avg_score": round(stats.avg_score, 4),
                    }
                    details.append(detail)

                    emit(
                        EdgePromoted(
                            source_id=src,
                            target_id=tgt,
                            hit_count=stats.hit_count,
                            avg_score=round(stats.avg_score, 4),
                        )
                    )

                    for hook in self._hooks.get("on_promote", []):
                        try:
                            hook(src, tgt, stats)
                        except Exception:
                            logger.debug("buffer.hook.failed", hook_event="on_promote")

                except Exception:
                    logger.warning("edge.promote.failed", source=src, target=tgt)

        for key in to_remove:
            del self._buffer[key]

        result = PromotionResult(
            promoted=promoted,
            remaining=len(self._buffer),
            details=details,
        )

        emit(
            BufferFlushed(
                promoted=promoted,
                remaining=len(self._buffer),
                total_promoted_lifetime=self._total_promoted,
                kg_coverage=None,
                timestamp=datetime.now(UTC).isoformat(),
            )
        )

        for hook in self._hooks.get("on_flush", []):
            try:
                hook(result)
            except Exception:
                logger.debug("buffer.hook.failed", hook_event="on_flush")

        return result

    def register_hook(self, event: str, callback: Any) -> None:
        """Register a lifecycle hook."""
        if event not in self._hooks:
            raise ValueError(f"Unknown hook event: {event}. Valid: {list(self._hooks)}")
        self._hooks[event].append(callback)

    def persist(self, path: Path | None = None) -> Path | None:
        """Write buffer to disk."""
        target = path or self._persistence_path
        if target is None:
            return None

        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)

        data = {}
        for (src, tgt), stats in self._buffer.items():
            key = f"{src}|{tgt}"
            data[key] = {
                "hit_count": stats.hit_count,
                "scores": stats.scores,
                "last_seen": stats.last_seen,
            }

        target.write_text(json.dumps(data, indent=2))
        return target

    @classmethod
    def load(cls, path: Path) -> EdgePromotionBuffer:
        """Load buffer from disk."""
        path = Path(path)
        buf = cls(path=path)

        if not path.exists():
            return buf

        try:
            data = json.loads(path.read_text())
            for key_str, stats_dict in data.items():
                parts = key_str.split("|", 1)
                if len(parts) != 2:
                    continue
                key = (parts[0], parts[1])
                buf._buffer[key] = EdgeStats(
                    hit_count=stats_dict.get("hit_count", 0),
                    scores=stats_dict.get("scores", []),
                    last_seen=stats_dict.get("last_seen", ""),
                )
        except (json.JSONDecodeError, TypeError):
            logger.warning("buffer.load.failed", path=str(path))

        return buf

    def summary(self) -> dict[str, Any]:
        """Summary stats for monitoring."""
        if not self._buffer:
            return {
                "buffered_edges": 0,
                "total_promoted": self._total_promoted,
                "ready_to_promote": 0,
            }

        hits = [s.hit_count for s in self._buffer.values()]
        return {
            "buffered_edges": len(self._buffer),
            "total_promoted": self._total_promoted,
            "max_hits": max(hits),
            "avg_hits": round(sum(hits) / len(hits), 1),
            "ready_to_promote": sum(
                1 for s in self._buffer.values() if s.hit_count >= 3 and s.avg_score >= 0.75
            ),
        }
