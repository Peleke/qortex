"""Log sinks: strategy pattern for event output destinations."""

from qortex.observe.sinks.base import LogSink
from qortex.observe.sinks.jsonl_sink import JsonlSink
from qortex.observe.sinks.noop_sink import NoOpSink
from qortex.observe.sinks.stdout_sink import StdoutSink

__all__ = ["LogSink", "JsonlSink", "StdoutSink", "NoOpSink"]
