"""Deterministic config snapshot hashing for reproducibility.

Every top-level operation (query, enrichment, ingest) can be correlated
with the exact configuration that produced it via a short hash.

Ported from TensorZero's config/snapshot.rs pattern (Blake3 -> Blake2b).
"""

from __future__ import annotations

import hashlib
import json


def config_snapshot_hash(
    rule_templates: dict | None = None,
    enrichment_config: dict | None = None,
    learner_configs: dict | None = None,
) -> str:
    """Blake2b-128 hash of deterministically-sorted config.

    All args are optional; only non-None configs are included in the hash.
    This means the hash changes only when relevant config changes.
    """
    payload_parts: dict = {}
    if rule_templates is not None:
        payload_parts["rules"] = rule_templates
    if enrichment_config is not None:
        payload_parts["enrichment"] = enrichment_config
    if learner_configs is not None:
        payload_parts["learners"] = learner_configs

    payload = json.dumps(payload_parts, sort_keys=True, default=str)
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()
