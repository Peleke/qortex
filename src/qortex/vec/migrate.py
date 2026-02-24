"""Vector migration: transfer all vectors from one backend to another.

Usage:
    from qortex.vec.migrate import migrate_vec
    result = await migrate_vec(source_index, dest_index, batch_size=500)

The function is idempotent — destination.add() is an upsert.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from qortex.observe.logging import get_logger
from qortex.observe.tracing import traced

if TYPE_CHECKING:
    from qortex.vec.index import VectorIndex

logger = get_logger(__name__)


@dataclass
class MigrateResult:
    """Result of a vector migration."""

    source_type: str
    dest_type: str
    batches: int
    vectors_read: int
    vectors_written: int
    duration_seconds: float
    dry_run: bool


@traced("vec.migrate")
async def migrate_vec(
    source: VectorIndex,
    destination: VectorIndex,
    *,
    batch_size: int = 500,
    dry_run: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> MigrateResult:
    """Migrate all vectors from source to destination.

    Idempotent — destination.add() is an upsert.

    Args:
        source: VectorIndex to read from.
        destination: VectorIndex to write to.
        batch_size: Number of vectors per batch.
        dry_run: If True, read but don't write.
        on_progress: Optional callback(batches_done, vectors_read).

    Returns:
        MigrateResult with counts and timing.
    """
    vectors_read = 0
    vectors_written = 0
    batches = 0
    t0 = time.monotonic()

    src_type = type(source).__name__
    dest_type = type(destination).__name__

    logger.info(
        "vec.migrate.start",
        source=src_type,
        dest=dest_type,
        batch_size=batch_size,
        dry_run=dry_run,
    )

    async for ids, embeddings in source.iter_all(batch_size=batch_size):
        vectors_read += len(ids)
        if not dry_run:
            await destination.add(ids, embeddings)
            vectors_written += len(ids)
        batches += 1
        logger.info(
            "vec.migrate.batch",
            batch=batches,
            vectors_read=vectors_read,
            vectors_written=vectors_written,
        )
        if on_progress:
            on_progress(batches, vectors_read)

    if not dry_run:
        await destination.persist()

    duration = time.monotonic() - t0
    result = MigrateResult(
        source_type=src_type,
        dest_type=dest_type,
        batches=batches,
        vectors_read=vectors_read,
        vectors_written=vectors_written,
        duration_seconds=round(duration, 3),
        dry_run=dry_run,
    )

    logger.info(
        "vec.migrate.complete",
        vectors_read=vectors_read,
        vectors_written=vectors_written,
        batches=batches,
        duration_seconds=result.duration_seconds,
        dry_run=dry_run,
    )

    return result
