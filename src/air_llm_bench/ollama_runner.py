"""Timed Ollama chat and embedding calls."""

from __future__ import annotations

import inspect
import time
from typing import Any

import ollama
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from air_llm_bench.metrics import timed

_NS = 1_000_000_000


def _ns(value: int | None) -> float | None:
    if not value:
        return None
    return value / _NS


def ensure_model(name: str) -> None:
    listed = ollama.list()
    have = set()
    for model in listed.models:
        have.add(getattr(model, "model", None) or getattr(model, "name", ""))
    if name in have:
        return
    print(f"pulling {name}", flush=True)
    ollama.pull(name)


def _chat(model: str, prompt: str, max_tokens: int) -> Any:
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }
    if "think" in inspect.signature(ollama.chat).parameters:
        kwargs["think"] = False
    else:
        kwargs["messages"] = [
            {"role": "user", "content": f"{prompt} /no_think"}
        ]
    return ollama.chat(**kwargs)


def run_chat(model: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    tracer = trace.get_tracer("air_llm_bench")
    started = time.perf_counter()
    with tracer.start_as_current_span(f"ollama.chat {model}") as span:
        span.set_attribute("openinference.span.kind", "LLM")
        span.set_attribute("langfuse.observation.type", "generation")
        span.set_attribute("gen_ai.system", "ollama")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("gen_ai.request.max_tokens", max_tokens)
        span.set_attribute("gen_ai.prompt", prompt)
        span.set_attribute("llm.model_name", model)
        span.set_attribute("llm.provider", "ollama")
        span.set_attribute("input.value", prompt)
        try:
            response, sampler = timed(
                lambda: _chat(model, prompt, max_tokens), ollama_model=model
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            return _error("ollama", model, "chat", started, exc)
        text = response.message.content or ""
        input_tokens = response.prompt_eval_count
        output_tokens = response.eval_count
        eval_seconds = _ns(response.eval_duration)
        load_seconds = _ns(response.load_duration)
        tokens_per_second = None
        if output_tokens and eval_seconds:
            tokens_per_second = output_tokens / eval_seconds
        span.set_attribute("gen_ai.response.model", response.model or model)
        span.set_attribute("gen_ai.completion", text)
        span.set_attribute("output.value", text)
        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
            span.set_attribute("llm.token_count.prompt", int(input_tokens))
        if output_tokens is not None:
            span.set_attribute("gen_ai.usage.output_tokens", int(output_tokens))
            span.set_attribute("llm.token_count.completion", int(output_tokens))
        span.set_status(Status(StatusCode.OK))
        return {
            "backend": "ollama",
            "model": model,
            "kind": "chat",
            "load_seconds": load_seconds,
            "generate_seconds": eval_seconds,
            "total_seconds": time.perf_counter() - started,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "tokens_per_second": tokens_per_second,
            "python_peak_rss_bytes": sampler.python_peak,
            "ollama_peak_rss_bytes": sampler.ollama_peak,
            "output": text,
            "error": None,
        }


def run_embed(model: str, prompt: str) -> dict[str, Any]:
    """nomic-embed-text is not covered by the Ollama instrumentor, so this span is manual."""
    tracer = trace.get_tracer("air_llm_bench")
    started = time.perf_counter()
    with tracer.start_as_current_span(f"ollama.embed {model}") as span:
        span.set_attribute("openinference.span.kind", "EMBEDDING")
        span.set_attribute("langfuse.observation.type", "embedding")
        span.set_attribute("gen_ai.system", "ollama")
        span.set_attribute("gen_ai.operation.name", "embeddings")
        span.set_attribute("gen_ai.request.model", model)
        span.set_attribute("llm.model_name", model)
        span.set_attribute("llm.provider", "ollama")
        span.set_attribute("input.value", prompt)
        span.set_attribute("embedding.model_name", model)
        try:
            response, sampler = timed(
                lambda: ollama.embed(model=model, input=prompt), ollama_model=model
            )
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            return _error("ollama", model, "embedding", started, exc)
        vectors = response.embeddings or []
        dimensions = len(vectors[0]) if vectors else 0
        input_tokens = getattr(response, "prompt_eval_count", None)
        span.set_attribute("embedding.embeddings_dimension", dimensions)
        span.set_attribute("gen_ai.response.model", response.model or model)
        span.set_attribute("output.value", f"{len(vectors)} vector(s), dim {dimensions}")
        if input_tokens is not None:
            span.set_attribute("gen_ai.usage.input_tokens", int(input_tokens))
        span.set_status(Status(StatusCode.OK))
        return {
            "backend": "ollama",
            "model": model,
            "kind": "embedding",
            "load_seconds": _ns(getattr(response, "load_duration", None)),
            "generate_seconds": _ns(getattr(response, "total_duration", None)),
            "total_seconds": time.perf_counter() - started,
            "input_tokens": input_tokens,
            "output_tokens": None,
            "tokens_per_second": None,
            "embedding_dimensions": dimensions,
            "python_peak_rss_bytes": sampler.python_peak,
            "ollama_peak_rss_bytes": sampler.ollama_peak,
            "output": f"{dimensions}-d embedding",
            "error": None,
        }


def _error(backend: str, model: str, kind: str, started: float, exc: Exception) -> dict[str, Any]:
    return {
        "backend": backend,
        "model": model,
        "kind": kind,
        "load_seconds": None,
        "generate_seconds": None,
        "total_seconds": time.perf_counter() - started,
        "input_tokens": None,
        "output_tokens": None,
        "tokens_per_second": None,
        "python_peak_rss_bytes": None,
        "ollama_peak_rss_bytes": None,
        "output": "",
        "error": f"{type(exc).__name__}: {exc}",
    }
