from __future__ import annotations

import concurrent.futures
import logging
import time

from chat_core import generate, init_history, load
from monitoring.prometheus_metrics import init_prometheus_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def _one_request(i: int, tokenizer, model) -> None:
    history = init_history()
    question = "Кратко объясни, что такое FMECA и зачем она нужна в промышленности?"
    answer = generate(
        tokenizer,
        model,
        history,
        question,
        do_sample=False,
        max_new_tokens=256,
    )
    print(f"[{i}] answer_len={len(answer)}")


def main() -> None:
    # Start Prometheus exporter on :8001
    init_prometheus_metrics(port=8001)

    # Warm-up model
    tokenizer, model = load()

    # Run a small burst of parallel requests to see concurrency on Prometheus graphs
    n_workers = 4
    n_requests = 12

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_one_request, i, tokenizer, model)
            for i in range(n_requests)
        ]
        for f in futures:
            f.result()
    total = time.perf_counter() - start
    print(f"Completed {n_requests} requests with {n_workers} workers in {total:.2f} s")


if __name__ == "__main__":
    main()