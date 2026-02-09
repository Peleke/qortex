"""Log sinks: strategy pattern for event output destinations."""

from qortex.observability.sinks.base import LogSink
from qortex.observability.sinks.jsonl_sink import JsonlSink
from qortex.observability.sinks.noop_sink import NoOpSink
from qortex.observability.sinks.stdout_sink import StdoutSink

__all__ = ["LogSink", "JsonlSink", "StdoutSink", "NoOpSink"]
