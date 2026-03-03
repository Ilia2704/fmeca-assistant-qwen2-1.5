from __future__ import annotations

import threading
import time

import psutil
from prometheus_client import Gauge, Histogram, start_http_server

from contextlib import contextmanager
from time import perf_counter
from typing import Any, Dict, Generator


# Histogram of end-to-end request latency (seconds)
REQUEST_LATENCY = Histogram(
    "fmeca_request_latency_seconds",
    "Latency of FMECA assistant requests",
    ["request_id", "backend", "scenario"],  # bounded label cardinality
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# Total tokens per request (prompt + completion)
REQUEST_TOKENS = Histogram(
    "fmeca_request_tokens_total",
    "Total tokens per request (prompt + completion)",
    ["request_id", "backend", "scenario"],
    buckets=(32, 64, 128, 256, 512, 1024, 2048),
)

# Effective throughput in tokens per second
TOKENS_PER_SECOND = Histogram(
    "fmeca_request_tokens_per_second",
    "Tokens per second during request",
    ["request_id", "backend", "scenario"],
    buckets=(5, 10, 20, 40, 80, 160, 320, 640, 1280),
)

REQUESTS_INFLIGHT = Gauge(
    "fmeca_requests_inflight",
    "Number of in-flight FMECA assistant requests",
    ["request_id", "backend", "scenario"],
)

# Basic process-level CPU / memory telemetry
PROCESS_CPU_PERCENT = Gauge(
    "fmeca_process_cpu_percent",
    "Process CPU usage percent (psutil.Process.cpu_percent())",
)

PROCESS_MEMORY_MB = Gauge(
    "fmeca_process_memory_mb",
    "Process resident memory size in megabytes",
)

_METRICS_STARTED = False


def init_prometheus_metrics(port: int = 8001) -> None:
    """Start Prometheus HTTP exporter and background sampler for CPU / RAM.

    Call this once at process start (for local demo you can do it in run_local.py).
    """
    global _METRICS_STARTED
    if _METRICS_STARTED:
        return
    _METRICS_STARTED = True

    # Start HTTP server for Prometheus to scrape, e.g. http://localhost:8001/metrics
    start_http_server(port)

    # Background thread to update CPU / RAM gauges once per second
    thread = threading.Thread(target=_collect_process_metrics_loop, daemon=True)
    thread.start()


def _collect_process_metrics_loop() -> None:
    proc = psutil.Process()
    # First call initializes internal measurement; second and onward give real values.
    proc.cpu_percent(interval=None)
    while True:
        cpu = proc.cpu_percent(interval=None)
        mem_mb = proc.memory_info().rss / (1024 * 1024)
        PROCESS_CPU_PERCENT.set(cpu)
        PROCESS_MEMORY_MB.set(mem_mb)
        time.sleep(1.0)


@contextmanager
def track_request(
    request_id: str,
    *,
    backend: str,
    scenario: str,
) -> Generator[Dict[str, Any], None, None]:
    """Track latency, inflight and token metrics for a single request."""
    labels = {
        "backend": backend,
        "scenario": scenario,
        "request_id": request_id,
    }
    REQUESTS_INFLIGHT.labels(**labels).inc()
    start = perf_counter()
    metrics: Dict[str, Any] = {}
    try:
        yield metrics
    finally:
        duration = perf_counter() - start
        REQUEST_LATENCY.labels(**labels).observe(duration)
        REQUESTS_INFLIGHT.labels(**labels).dec()

        prompt_tokens = metrics.get("prompt_tokens")
        completion_tokens = metrics.get("completion_tokens")
        if prompt_tokens is not None and completion_tokens is not None:
            total_tokens = int(prompt_tokens) + int(completion_tokens)
            if total_tokens > 0:
                REQUEST_TOKENS.labels(**labels).observe(total_tokens)
                if duration > 0:
                    TOKENS_PER_SECOND.labels(**labels).observe(
                        float(total_tokens) / float(duration)
                    )