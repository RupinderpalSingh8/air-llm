# AirLLM vs Ollama

Same short prompt, two local runtimes, one OpenTelemetry trace.

Ollama keeps a quantized GGUF model resident and decodes it with llama.cpp. AirLLM leaves full-precision weights on disk and loads one transformer layer at a time for every token. On this CPU-only machine that makes AirLLM slower even for a 0.6B model. The trace is the comparison: Ollama chat spans, one embedding span, and an AirLLM span with `airllm.load`, `airllm.tokenize`, and `airllm.forward` children.

This machine has no NVIDIA GPU, so AirLLM runs with `device="cpu"` and no 4-bit or 8-bit compression. The model is `Qwen/Qwen3-0.6B`. Ollama runs `qwen3:0.6b` (same size, Q4), the existing `qwen3:latest` (8.2B Q4), and one `nomic-embed-text` embedding.

## Setup

```bash
uv sync
cp .env.example .env   # then fill the keys, or use the generated .env
docker compose up -d   # Langfuse only
uv run compare-llms
```

Langfuse is at http://127.0.0.1:3000. Sign in with `LANGFUSE_INIT_USER_EMAIL` and `LANGFUSE_INIT_USER_PASSWORD` from `.env`. The run writes `results/compare.json` with timings, token counts, and the trace id.

Weights and AirLLM layer shards go under `.cache/` on this disk.

## Switching the tracer

`.env` is the only switch:

| `TRACE_BACKEND` | Where spans go |
| --- | --- |
| `langfuse` | http://127.0.0.1:3000/api/public/otel (default, started by `docker compose up`) |
| `grafana` | Tempo at http://127.0.0.1:4318/v1/traces, UI http://127.0.0.1:3001 |
| `console` | Terminal only |

`TRACE_ALSO_CONSOLE=true` still prints spans when a collector is selected.

Grafana and Tempo are in `docker-compose.grafana.yml` and `grafana/`. Nothing in this setup starts them. To use that backend later, stop other listeners on ports 4317 and 4318, then:

```bash
docker compose -f docker-compose.grafana.yml up -d
```

Set `TRACE_BACKEND=grafana` and run `uv run compare-llms` again.

## What the numbers mean

`load_seconds` for Ollama is the server-reported model load. For AirLLM it is `from_pretrained`, which downloads the checkpoint on the first run and writes layer shards. `tokens_per_second` is decode speed after that load. `ollama_peak_rss_bytes` is the resident size of that one model from `ollama ps` (the weights sit in `llama-server`, not in this Python process). `python_peak_rss_bytes` is this process, which is where AirLLM holds the active layer.
