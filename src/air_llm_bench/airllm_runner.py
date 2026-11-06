"""Timed AirLLM generation on CPU. Layers are streamed from disk, one at a time."""

from __future__ import annotations

import os
import time
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from air_llm_bench.metrics import timed


def _set_llm(span, model: str, prompt: str, max_tokens: int) -> None:
    span.set_attribute("openinference.span.kind", "LLM")
    span.set_attribute("langfuse.observation.type", "generation")
    span.set_attribute("gen_ai.system", "airllm")
    span.set_attribute("gen_ai.operation.name", "chat")
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.request.max_tokens", max_tokens)
    span.set_attribute("gen_ai.prompt", prompt)
    span.set_attribute("llm.model_name", model)
    span.set_attribute("llm.provider", "airllm")
    span.set_attribute("input.value", prompt)
    span.set_attribute("llm.invocation_parameters", '{"device":"cpu","compression":null}')


def run_airllm(model_id: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    tracer = trace.get_tracer("air_llm_bench")
    started = time.perf_counter()
    shard_dir = os.environ["AIRLLM_SHARD_DIR"]
    os.makedirs(shard_dir, exist_ok=True)
    with tracer.start_as_current_span(f"airllm.generate {model_id}") as span:
        _set_llm(span, model_id, prompt, max_tokens)
        try:
            from airllm import AutoModel

            with tracer.start_as_current_span("airllm.load") as load_span:
                load_span.set_attribute("airllm.model", model_id)
                load_started = time.perf_counter()
                model, load_sampler = timed(
                    lambda: AutoModel.from_pretrained(
                        model_id,
                        device="cpu",
                        compression=None,
                        layer_shards_saving_path=shard_dir,
                        prefetching=True,
                    )
                )
                load_seconds = time.perf_counter() - load_started
                load_span.set_attribute("airllm.load_seconds", load_seconds)
                load_span.set_attribute("python.rss_bytes", load_sampler.python_peak)

            with tracer.start_as_current_span("airllm.tokenize"):
                messages = [{"role": "user", "content": prompt}]
                try:
                    text = model.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except TypeError:
                    text = model.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                encoded = model.tokenizer(
                    [text],
                    return_tensors="pt",
                    return_attention_mask=False,
                    truncation=True,
                    max_length=256,
                    padding=False,
                )
                input_ids = encoded["input_ids"].to(model.device)

            with tracer.start_as_current_span("airllm.forward"):
                gen_started = time.perf_counter()
                output, gen_sampler = timed(
                    lambda: model.generate(
                        input_ids,
                        max_new_tokens=max_tokens,
                        use_cache=True,
                        return_dict_in_generate=True,
                        do_sample=False,
                    )
                )
                generate_seconds = time.perf_counter() - gen_started

            sequences = output.sequences
            input_tokens = int(input_ids.shape[-1])
            output_tokens = int(sequences.shape[-1] - input_tokens)
            new_ids = sequences[0][input_tokens:]
            completion = model.tokenizer.decode(new_ids, skip_special_tokens=True)
            tokens_per_second = (
                output_tokens / generate_seconds if generate_seconds else None
            )
            span.set_attribute("gen_ai.response.model", model_id)
            span.set_attribute("gen_ai.completion", completion)
            span.set_attribute("output.value", completion)
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            span.set_attribute("llm.token_count.prompt", input_tokens)
            span.set_attribute("llm.token_count.completion", output_tokens)
            span.set_status(Status(StatusCode.OK))
            python_peak = max(load_sampler.python_peak, gen_sampler.python_peak)
            return {
                "backend": "airllm",
                "model": model_id,
                "kind": "chat",
                "device": "cpu",
                "precision": "checkpoint dtype, layers streamed from disk",
                "load_seconds": load_seconds,
                "generate_seconds": generate_seconds,
                "total_seconds": time.perf_counter() - started,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_second": tokens_per_second,
                "python_peak_rss_bytes": python_peak,
                "ollama_peak_rss_bytes": None,
                "output": completion,
                "error": None,
            }
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            return {
                "backend": "airllm",
                "model": model_id,
                "kind": "chat",
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
