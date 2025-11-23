"""OpenTelemetry tracing helper for NexusZero Optimizer.

This module provides a thin wrapper to initialize and expose a tracer. It
keeps a no-op fallback if OpenTelemetry isn't installed to preserve testability.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SpanExportResult,
    )
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path in environments without OTEL deps
    OTEL_AVAILABLE = False


DEFAULT_SERVICE_NAME = os.environ.get("NEXUSZERO_SERVICE", "nexuszero-optimizer")


def init_tracer(service_name: Optional[str] = None, export_to_console: Optional[bool] = None):
    """Initialize an OpenTelemetry TracerProvider with either OTLP or Console exporter.

    This function is idempotent and safe to call multiple times.
    It also tolerates missing OpenTelemetry packages and falls back to a no-op tracer.

    Args:
        service_name: Optional name for the service.
        export_to_console: If True, always add a ConsoleSpanExporter for debugging.
    """
    if not OTEL_AVAILABLE:
        logger.debug("OpenTelemetry packages are not installed; tracing disabled.")
        return

    if export_to_console is None:
        _v = os.environ.get("NEXUSZERO_OTEL_CONSOLE", "0")
        export_to_console = _v in ("1", "true", "True")

    if service_name is None:
        service_name = DEFAULT_SERVICE_NAME

    # Create resource
    resource = Resource.create({"service.name": service_name})

    # Initialize provider
    provider = TracerProvider(resource=resource)

    # Configure OTLP exporter when endpoint is available
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint:
        try:
            otlp_exporter = OTLPSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            logger.debug("Initialized OTLPSpanExporter for tracing.")
        except Exception as ex:  # pragma: no cover - environmental error
            logger.warning("Failed to initialize OTLP exporter: %s", ex)

    if export_to_console:
        # Wrap the ConsoleSpanExporter to be resilient in testing environments
        # where stdout/stderr may be captured or closed by the runner (pytest).
        class SafeConsoleSpanExporter(ConsoleSpanExporter):
            def export(self, spans):
                try:
                    return super().export(spans)
                except Exception:  # pragma: no cover - environment-specific
                    logger.warning(
                        "ConsoleSpanExporter failed to write span; ignoring"
                    )
                    return SpanExportResult.SUCCESS

        processor = BatchSpanProcessor(SafeConsoleSpanExporter())
        provider.add_span_processor(processor)
        logger.debug("Initialized ConsoleSpanExporter for tracing.")

    trace.set_tracer_provider(provider)


def get_tracer(name: Optional[str] = None):
    """Return an OpenTelemetry tracer instance or a no-op fallback.

    Usage:
        from nexuszero_optimizer.utils.tracing import get_tracer, init_tracer
        init_tracer()  # once at process start
        tracer = get_tracer(__name__)

    """
    if not OTEL_AVAILABLE:
        # Return a shim tracer-like object with context manager methods
        class NoopSpan:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def set_attribute(self, *args, **kwargs):
                return None

            def add_event(self, *args, **kwargs):
                return None

        class NoopTracer:
            def start_as_current_span(self, *args, **kwargs):
                return NoopSpan()

        return NoopTracer()
    if name is None:
        name = DEFAULT_SERVICE_NAME
    return trace.get_tracer(name)


def span(name: str):
    """Function decorator to create a span around the function call.

    Usage:
        @span("train_epoch")
        def train_epoch(...):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            t = get_tracer(func.__module__)
            with t.start_as_current_span(name):
                return func(*args, **kwargs)

        wrapper.__name__ = f"traced_{func.__name__}"
        return wrapper

    return decorator
