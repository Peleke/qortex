"""Create OTel instruments from the declarative metric schema.

Generates instruments and Views so custom histogram buckets apply to both
OTLP push and PrometheusMetricReader pull paths.
"""

from __future__ import annotations

from typing import Any

from qortex_observe.metrics_schema import METRICS, MetricType


def create_instruments(meter: Any) -> dict[str, Any]:
    """Create OTel instruments from METRICS schema.

    Returns {metric_name: instrument} dict. Instruments have .add() (counter),
    .record() (histogram), or .set() (gauge) methods.
    """
    instruments: dict[str, Any] = {}

    for m in METRICS:
        if m.type == MetricType.COUNTER:
            instruments[m.name] = meter.create_counter(
                m.name, description=m.description,
            )
        elif m.type == MetricType.HISTOGRAM:
            instruments[m.name] = meter.create_histogram(
                m.name, description=m.description,
            )
        elif m.type == MetricType.GAUGE:
            instruments[m.name] = meter.create_gauge(
                m.name, description=m.description,
            )

    return instruments


def create_views() -> list[Any]:
    """Create OTel Views for histograms with custom bucket boundaries.

    These Views are passed to MeterProvider so both OTLP and
    PrometheusMetricReader get the correct bucket configurations.
    Without Views, histograms use OTel's default buckets.
    """
    from opentelemetry.sdk.metrics.view import (
        ExplicitBucketHistogramAggregation,
        View,
    )

    views = []
    for m in METRICS:
        if m.type == MetricType.HISTOGRAM and m.buckets:
            views.append(
                View(
                    instrument_name=m.name,
                    aggregation=ExplicitBucketHistogramAggregation(
                        boundaries=list(m.buckets),
                    ),
                )
            )

    return views
