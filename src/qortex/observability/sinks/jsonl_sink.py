"""JSONL file sink: append JSON lines to a file. Loki-ready format."""

from __future__ import annotations

import json
from pathlib import Path


class JsonlSink:
    """Append JSON lines to a file. Thread-safe via append mode."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_dict: dict) -> None:
        line = json.dumps(event_dict, default=str) + "\n"
        with open(self._path, "a") as f:
            f.write(line)
