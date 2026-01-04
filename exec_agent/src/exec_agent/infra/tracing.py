"""Trace helpers for tool execution."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace


@contextmanager
def tool_span(name: str, **attributes: object) -> Iterator[None]:
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield
