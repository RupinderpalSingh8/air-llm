"""Run the same prompt through Ollama and AirLLM and write results/compare.json."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from opentelemetry.trace import Status, StatusCode

ROOT = Path(__file__).resolve().parents[2]


def _prepare_env() -> None:
    load_dotenv(ROOT / ".env")
    cache = ROOT / ".cache"
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("AIRLLM_SHARD_DIR", str(cache / "airllm-shards"))
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    os.makedirs(os.environ["AIRLLM_SHARD_DIR"], exist_ok=True)


def main() -> None:
    _prepare_env()
    from air_llm_bench.airllm_runner import run_airllm
    from air_llm_bench.ollama_runner import ensure_model, run_chat, run_embed
    from air_llm_bench.tracing import setup_tracing, shutdown_tracing

    prompt = os.environ.get(
        "COMPARE_PROMPT",
        "What is the capital of France? Answer in one short sentence.",
    )
    max_tokens = int(os.environ.get("MAX_NEW_TOKENS", "32"))
    ollama_small = os.environ.get("OLLAMA_SMALL_MODEL", "qwen3:0.6b")
    ollama_large = os.environ.get("OLLAMA_LARGE_MODEL", "qwen3:latest")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    air_model = os.environ.get("AIRLLM_MODEL", "Qwen/Qwen3-0.6B")

    tracer, target = setup_tracing()
    runs: list[dict] = []
    trace_id = ""
    try:
        with tracer.start_as_current_span("compare.airllm_vs_ollama") as span:
            span.set_attribute("langfuse.observation.type", "span")
            span.set_attribute("gen_ai.operation.name", "compare")
            span.set_attribute("compare.prompt", prompt)
            span.set_attribute("compare.max_new_tokens", max_tokens)
            trace_id = format(span.get_span_context().trace_id, "032x")
            print(f"trace_id={trace_id}", flush=True)

            for model in (ollama_small, ollama_large, embed_model):
                ensure_model(model)

            print(f"ollama chat {ollama_small}", flush=True)
            runs.append(run_chat(ollama_small, prompt, max_tokens))
            print(f"ollama chat {ollama_large}", flush=True)
            runs.append(run_chat(ollama_large, prompt, max_tokens))
            print(f"ollama embed {embed_model}", flush=True)
            runs.append(run_embed(embed_model, prompt))
            print(f"airllm {air_model} (CPU, first run downloads and splits)", flush=True)
            runs.append(run_airllm(air_model, prompt, max_tokens))

            failed = [run for run in runs if run.get("error")]
            if failed:
                span.set_status(Status(StatusCode.ERROR, failed[0]["error"]))
            else:
                span.set_status(Status(StatusCode.OK))
    finally:
        shutdown_tracing()

    trace_url = f"{target.ui_url}/traces/{trace_id}" if target.ui_url and trace_id else target.ui_url
    report = {
        "trace_id": trace_id,
        "trace_backend": target.backend,
        "ui_url": target.ui_url,
        "trace_url": trace_url,
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "difference": {
            "ollama": "Quantized GGUF stays resident. llama.cpp decodes on CPU.",
            "airllm": "Full-precision weights stay on disk. Each token reloads every transformer layer onto CPU, then drops it.",
        },
        "runs": runs,
    }
    out = ROOT / "results" / "compare.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}", flush=True)
    if target.ui_url:
        print(f"ui {target.ui_url}", flush=True)
        print(f"trace {trace_url}", flush=True)
    for run in runs:
        status = run.get("error") or "ok"
        print(
            f"{run['backend']:8} {run['kind']:10} {run['model']:22} "
            f"total={run['total_seconds']:.2f}s tokens/s={run.get('tokens_per_second')} {status}",
            flush=True,
        )


if __name__ == "__main__":
    main()
