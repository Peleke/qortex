"""Stdout sink: write JSON to stdout (dev mode)."""

from __future__ import annotations

import json
import sys


class StdoutSink:
    """Write JSON event dicts to stdout."""

    def write(self, event_dict: dict) -> None:
        line = json.dumps(event_dict, default=str)
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
