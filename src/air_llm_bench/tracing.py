"""OpenTelemetry setup. TRACE_BACKEND selects langfuse, grafana, or console."""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openinference.instrumentation.ollama import OllamaInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Tracer

_PROVIDER: TracerProvider | None = None


@dataclass(frozen=True)
class TraceTarget:
    backend: str
    ui_url: str
    traces_endpoint: str | None


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _traces_url(base: str) -> str:
    endpoint = base.strip().rstrip("/")
    if endpoint.endswith("/v1/traces"):
        return endpoint
    return endpoint + "/v1/traces"


def resolve_target() -> TraceTarget:
    backend = os.environ.get("TRACE_BACKEND", "langfuse").strip().lower()
    if backend == "console":
        return TraceTarget(backend, "", None)
    if backend == "langfuse":
        host = os.environ.get("LANGFUSE_HOST", "http://127.0.0.1:3000").rstrip("/")
        base = os.environ.get("LANGFUSE_OTLP_ENDPOINT", f"{host}/api/public/otel")
        project = os.environ.get("LANGFUSE_PROJECT_ID", "")
        ui = f"{host}/project/{project}" if project else host
        return TraceTarget(backend, ui, _traces_url(base))
    if backend == "grafana":
        base = os.environ.get("GRAFANA_OTLP_ENDPOINT", "http://127.0.0.1:4318")
        ui = os.environ.get("GRAFANA_UI_URL", "http://127.0.0.1:3001")
        return TraceTarget(backend, ui, _traces_url(base))
    raise SystemExit(
        f"Unknown TRACE_BACKEND={backend!r}. Use langfuse, grafana, or console."
    )


def _langfuse_headers() -> dict[str, str]:
    public = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    if not public or not secret:
        raise SystemExit(
            "TRACE_BACKEND=langfuse needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env."
        )
    token = base64.b64encode(f"{public}:{secret}".encode()).decode()
    return {
        "Authorization": f"Basic {token}",
        "x-langfuse-ingestion-version": "4",
    }


def setup_tracing() -> tuple[Tracer, TraceTarget]:
    """Install the global tracer and the Ollama chat instrumentor."""
    global _PROVIDER
    load_dotenv()
    target = resolve_target()
    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": os.environ.get("OTEL_SERVICE_NAME", "air-llm-bench"),
            }
        )
    )
    if _flag("TRACE_ALSO_CONSOLE", "true"):
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    if target.traces_endpoint:
        headers = _langfuse_headers() if target.backend == "langfuse" else None
        exporter = OTLPSpanExporter(endpoint=target.traces_endpoint, headers=headers)
        provider.add_span_processor(
            BatchSpanProcessor(exporter, schedule_delay_millis=1000)
        )
    trace.set_tracer_provider(provider)
    OllamaInstrumentor().instrument(tracer_provider=provider)
    _PROVIDER = provider
    return trace.get_tracer("air_llm_bench"), target


def shutdown_tracing() -> None:
    if _PROVIDER is not None:
        _PROVIDER.force_flush(timeout_millis=30000)
        _PROVIDER.shutdown()
