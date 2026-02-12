"""Persistence for arm state: JSON file store with context partitioning."""

from __future__ import annotations

import json
from pathlib import Path

from qortex.learning.types import ArmState, context_hash


class ArmStateStore:
    """Persists arm states as JSON, partitioned by context hash.

    File layout: {state_dir}/{learner_name}.json
    JSON structure: { context_hash: { arm_id: ArmState.to_dict() } }
    """

    def __init__(self, learner_name: str, state_dir: str = "") -> None:
        self._name = learner_name
        if state_dir:
            self._dir = Path(state_dir)
        else:
            self._dir = Path("~/.qortex/learning").expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"{learner_name}.json"
        self._data: dict[str, dict[str, ArmState]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            raw = json.loads(self._path.read_text())
            for ctx, arms in raw.items():
                self._data[ctx] = {
                    arm_id: ArmState.from_dict(state)
                    for arm_id, state in arms.items()
                }

    def save(self) -> None:
        out: dict[str, dict[str, dict]] = {}
        for ctx, arms in self._data.items():
            out[ctx] = {arm_id: state.to_dict() for arm_id, state in arms.items()}
        self._path.write_text(json.dumps(out, indent=2))

    def get(self, arm_id: str, context: dict | None = None) -> ArmState:
        ctx = context_hash(context or {})
        return self._data.get(ctx, {}).get(arm_id, ArmState())

    def get_all(self, context: dict | None = None) -> dict[str, ArmState]:
        ctx = context_hash(context or {})
        return dict(self._data.get(ctx, {}))

    def put(self, arm_id: str, state: ArmState, context: dict | None = None) -> None:
        ctx = context_hash(context or {})
        if ctx not in self._data:
            self._data[ctx] = {}
        self._data[ctx][arm_id] = state

    def get_all_contexts(self) -> list[str]:
        return list(self._data.keys())
