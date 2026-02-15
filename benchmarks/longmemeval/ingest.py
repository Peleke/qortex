"""Session ingestion adapter for LongMemEval.

Feeds timestamped haystack sessions into the qortex knowledge graph.
Each session becomes concept nodes with temporal metadata and edges
linking co-occurring entities across sessions.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from qortex.client import LocalQortexClient

logger = logging.getLogger(__name__)


@dataclass
class SessionTurn:
    """A single turn in a conversation session."""

    role: str  # "user" or "assistant"
    content: str
    has_answer: bool = False


@dataclass
class IngestStats:
    """Statistics from ingesting sessions for a single question."""

    sessions_ingested: int = 0
    concepts_created: int = 0
    edges_created: int = 0
    rules_created: int = 0
    warnings: list[str] = field(default_factory=list)


def ingest_sessions(
    client: LocalQortexClient,
    sessions: list[list[dict[str, Any]]],
    session_dates: list[str],
    domain: str = "longmemeval",
) -> IngestStats:
    """Ingest LongMemEval haystack sessions into qortex.

    Each session is a list of turns (user/assistant messages). Sessions
    are timestamped via session_dates. We ingest each session as a
    single text block with temporal metadata embedded, so the LLM
    extraction pipeline can capture temporal relationships.

    Args:
        client: LocalQortexClient instance.
        sessions: List of sessions, each a list of turn dicts with
            'role' and 'content' keys.
        session_dates: ISO date strings, one per session.
        domain: Domain name for the ingested data.

    Returns:
        IngestStats with counts of created entities.
    """
    stats = IngestStats()

    for i, (session, date) in enumerate(zip(sessions, session_dates)):
        # Format session as a timestamped conversation block
        lines = [f"[Session date: {date}]"]
        for turn in session:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if not content.strip():
                continue
            lines.append(f"{role}: {content}")

        session_text = "\n".join(lines)
        if not session_text.strip():
            continue

        session_name = f"session_{i:04d}_{date}"

        try:
            result = client.ingest_text(
                text=session_text,
                domain=domain,
                format="text",
                name=session_name,
            )
            stats.sessions_ingested += 1
            stats.concepts_created += result.concepts
            stats.edges_created += result.edges
            stats.rules_created += result.rules
            stats.warnings.extend(result.warnings)
        except Exception as e:
            warning = f"Failed to ingest session {i} ({date}): {e}"
            logger.warning(warning)
            stats.warnings.append(warning)

    return stats


def ingest_sessions_structured(
    client: LocalQortexClient,
    sessions: list[list[dict[str, Any]]],
    session_dates: list[str],
    domain: str = "longmemeval",
) -> IngestStats:
    """Ingest sessions as structured concepts (no LLM extraction).

    Faster alternative to ingest_sessions() that skips LLM extraction.
    Each session turn becomes a concept node directly, with edges
    linking sequential turns within a session and temporal metadata.

    Use this for quick benchmarks where LLM extraction cost is prohibitive
    (500 questions x ~40 sessions each = ~20K LLM calls with full extraction).

    Args:
        client: LocalQortexClient instance.
        sessions: List of sessions.
        session_dates: ISO date strings, one per session.
        domain: Domain name.

    Returns:
        IngestStats with counts.
    """
    stats = IngestStats()

    all_concepts: list[dict[str, Any]] = []
    all_edges: list[dict[str, Any]] = []

    for sess_idx, (session, date) in enumerate(zip(sessions, session_dates)):
        prev_concept_id = None

        for turn_idx, turn in enumerate(session):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            if not content.strip():
                continue

            # Create a stable ID from session index + turn index
            raw_id = f"{domain}:s{sess_idx:04d}:t{turn_idx:03d}"
            concept_id = f"{domain}:{hashlib.sha256(raw_id.encode()).hexdigest()[:12]}"

            # Truncate name to 80 chars for readability
            name = f"[{date}] {role}: {content[:60]}"
            description = f"[Session {sess_idx}, {date}] {role}: {content}"

            all_concepts.append({
                "id": concept_id,
                "name": name,
                "description": description,
                "properties": {
                    "session_idx": sess_idx,
                    "turn_idx": turn_idx,
                    "session_date": date,
                    "role": role,
                    "has_answer": turn.get("has_answer", False),
                },
            })

            # Sequential edge within session
            if prev_concept_id is not None:
                all_edges.append({
                    "source": prev_concept_id,
                    "target": concept_id,
                    "relation_type": "part_of",
                })
            prev_concept_id = concept_id

    if not all_concepts:
        return stats

    # Ingest in batches to avoid memory issues
    batch_size = 500
    for batch_start in range(0, len(all_concepts), batch_size):
        batch_concepts = all_concepts[batch_start : batch_start + batch_size]

        # Collect edges that reference concepts in this batch
        batch_concept_ids = {c["id"] for c in batch_concepts}
        # Also need IDs from previous batch for cross-batch edges
        batch_edges = [
            e
            for e in all_edges
            if e["source"] in batch_concept_ids or e["target"] in batch_concept_ids
        ]

        try:
            result = client.ingest_structured(
                concepts=batch_concepts,
                domain=domain,
                edges=batch_edges,
            )
            stats.concepts_created += result.concepts
            stats.edges_created += result.edges
            stats.rules_created += result.rules
            stats.warnings.extend(result.warnings)
        except Exception as e:
            warning = f"Failed to ingest batch at {batch_start}: {e}"
            logger.warning(warning)
            stats.warnings.append(warning)

    stats.sessions_ingested = len(sessions)
    return stats
