from __future__ import annotations

import threading
import time

import psutil
from prometheus_client import Gauge, Histogram, start_http_server

# Histogram of end-to-end request latency (seconds)
REQUEST_LATENCY = Histogram(
    "fmeca_request_latency_seconds",
    "Latency of FMECA assistant requests",
    buckets=(0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
)

# How many requests are in-flight right now
REQUESTS_INFLIGHT = Gauge(
    "fmeca_requests_inflight",
    "Number of in-flight FMECA assistant requests",
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


from contextlib import contextmanager
from time import perf_counter


@contextmanager
def track_request():
    """Context manager for per-request latency / in-flight metrics.

    Example:
        with track_request():
            ... do work ...
    """
    REQUESTS_INFLIGHT.inc()
    start = perf_counter()
    try:
        yield
    finally:
        duration = perf_counter() - start
        REQUEST_LATENCY.observe(duration)
        REQUESTS_INFLIGHT.dec()