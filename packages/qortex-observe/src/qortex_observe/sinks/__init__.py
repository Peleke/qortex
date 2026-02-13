"""Log sinks: strategy pattern for event output destinations."""

from qortex_observe.sinks.base import LogSink
from qortex_observe.sinks.jsonl_sink import JsonlSink
from qortex_observe.sinks.noop_sink import NoOpSink
from qortex_observe.sinks.stdout_sink import StdoutSink

__all__ = ["LogSink", "JsonlSink", "StdoutSink", "NoOpSink"]
