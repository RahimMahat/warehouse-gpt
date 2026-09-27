"""Tracing (OpenTelemetry, OpenInference conventions) and structured logging.

Tracing is off by default and costs nothing when off: spans go to OpenTelemetry's no-op tracer.
With ``WGPT_TRACING=true`` spans are exported over OTLP/HTTP to Arize Phoenix (or any OTLP
collector). Attribute names follow OpenInference, so Phoenix renders LLM spans with prompts,
completions and token counts, retrieval spans with documents, and guard spans as guardrails:

    uv run --extra phoenix phoenix serve      # UI at http://localhost:6006
    WGPT_TRACING=true uv run wgpt ask "..."
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode

from warehouse_gpt.config import Settings

TRACER_NAME = "warehouse_gpt"
_initialized = False

# OpenInference span kinds and attribute keys (kept local to avoid another dependency).
KIND = "openinference.span.kind"
INPUT, OUTPUT = "input.value", "output.value"
AGENT, CHAIN, LLM, RETRIEVER, TOOL, GUARDRAIL = "AGENT", "CHAIN", "LLM", "RETRIEVER", "TOOL", "GUARDRAIL"


def init_tracing(settings: Settings, exporter: Any | None = None) -> bool:
    """Install a tracer provider once. ``exporter`` overrides OTLP (tests pass an in-memory one)."""
    global _initialized
    if _initialized or not (settings.tracing or exporter is not None):
        return _initialized
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

    resource = Resource.create(
        {"service.name": "warehouse-gpt", "openinference.project.name": "warehouse-gpt"}
    )
    provider = TracerProvider(resource=resource)
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=settings.otlp_endpoint)))
    trace.set_tracer_provider(provider)
    _initialized = True
    return True


def tracer() -> trace.Tracer:
    return trace.get_tracer(TRACER_NAME)


def _attr(value: Any) -> Any:
    if isinstance(value, str | bool | int | float):
        return value
    if isinstance(value, Sequence) and all(isinstance(v, str | bool | int | float) for v in value):
        return list(value)
    return json.dumps(value, default=str)


@contextmanager
def span(name: str, kind: str, input: Any = None, **attributes: Any) -> Iterator[Span]:
    """Start a span with an OpenInference kind; exceptions mark it as an error and re-raise."""
    with tracer().start_as_current_span(name) as s:
        s.set_attribute(KIND, kind)
        if input is not None:
            s.set_attribute(INPUT, _attr(input))
        for k, v in attributes.items():
            if v is not None:
                s.set_attribute(k, _attr(v))
        try:
            yield s
        except Exception as exc:
            s.record_exception(exc)
            s.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
            raise


def set_output(s: Span, output: Any, **attributes: Any) -> None:
    if output is not None:
        s.set_attribute(OUTPUT, _attr(output))
    for k, v in attributes.items():
        if v is not None:
            s.set_attribute(k, _attr(v))


def llm_attributes(
    model: str,
    messages: Sequence[Mapping[str, str]],
    params: Mapping[str, Any],
) -> dict[str, Any]:
    attrs: dict[str, Any] = {"llm.model_name": model, "llm.invocation_parameters": json.dumps(params)}
    for i, m in enumerate(messages):
        attrs[f"llm.input_messages.{i}.message.role"] = m["role"]
        attrs[f"llm.input_messages.{i}.message.content"] = m["content"]
    return attrs


# -- logging -------------------------------------------------------------------------------
def configure_logging(level: str = "WARNING", json_logs: bool = False) -> None:
    """structlog setup shared by CLI (console, quiet) and API (JSON, one event per request)."""
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderer: Any = structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[*processors, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelNamesMapping()[level.upper()]),
        cache_logger_on_first_use=False,
    )
