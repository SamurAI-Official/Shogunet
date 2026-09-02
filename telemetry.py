"""
Shogunet telemetry hooks
========================

Dependency-free stand-in mirroring ShugoCore's ``telemetry.get_tracer``
contract: callers ask for a tracer once at import time and store spans only
if OpenTelemetry is installed. Without it, every call degrades to a no-op
context manager -- the hot path never pays a telemetry tax.
"""

import contextlib
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_NOOP_SPAN = contextlib.nullcontext()


def get_tracer(name: str = "shugonet"):
    """Return an OTel-compatible tracer, or a no-op stand-in.

    The stand-in exposes ``start_as_current_span(...)`` as a no-op context
    manager and ``add_event``-free spans, so instrumented code runs identically
    with or without OpenTelemetry.
    """
    try:
        from opentelemetry import trace  # type: ignore
        return trace.get_tracer(name)
    except Exception:
        pass

    class _NoopSpan(contextlib.AbstractContextManager):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def set_attribute(self, key: str, value: Any) -> None:
            pass

        def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
            pass

    class _NoopTracer:
        def start_as_current_span(self, span_name: str, *args, **kwargs):
            return _NoopSpan()

        def start_span(self, span_name: str, *args, **kwargs):
            return _NoopSpan()

    return _NoopTracer()