"""Timing and memory samples shared by the runners."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

import psutil

T = TypeVar("T")


def _same_model(requested: str, loaded: str) -> bool:
    if requested == loaded:
        return True
    # `ollama ps` reports an implicit tag as `:latest`.
    if ":" not in requested and loaded == f"{requested}:latest":
        return True
    if requested.endswith(":latest") and loaded == requested.split(":", 1)[0]:
        return True
    return False


def ollama_model_bytes(model_name: str) -> int:
    """Resident size of one loaded model, from `ollama ps`.

    The weights live in a llama-server process, not the Python client and not
    the small `ollama serve` process.
    """
    import ollama

    try:
        running = ollama.ps()
    except Exception:
        return 0
    for item in running.models:
        name = item.name or item.model or ""
        if _same_model(model_name, name):
            return int(item.size or 0)
    return 0


def python_rss() -> int:
    return psutil.Process().memory_info().rss


class PeakSampler:
    """Sample this process, and one Ollama model's resident size, during a call."""

    def __init__(self, ollama_model: str | None = None) -> None:
        self.ollama_model = ollama_model
        self.python_peak = python_rss()
        self.ollama_peak = ollama_model_bytes(ollama_model) if ollama_model else 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.python_peak = max(self.python_peak, python_rss())
            if self.ollama_model:
                self.ollama_peak = max(self.ollama_peak, ollama_model_bytes(self.ollama_model))
            self._stop.wait(0.5)

    def __enter__(self) -> PeakSampler:
        self._thread.start()
        return self

    def __exit__(self, *_) -> None:
        self._stop.set()
        self._thread.join(timeout=1)


def timed(fn: Callable[[], T], ollama_model: str | None = None) -> tuple[T, PeakSampler]:
    with PeakSampler(ollama_model) as sampler:
        return fn(), sampler
