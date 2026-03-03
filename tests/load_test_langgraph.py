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

log = logging.getLogger("load_test_langgraph")

SCENARIOS = [
    (
        "pump_cavitation",
        "Сделай краткий FMECA-анализ кавитации центробежного насоса: "
        "основные функции, режим кавитации, последствия и меры контроля.",
    ),
    (
        "seal_leak",
        "Сделай краткий FMECA-анализ утечки через торцевое уплотнение насоса: "
        "возможные причины, последствия и рекомендуемые действия.",
    ),
    (
        "bearing_failure",
        "Сделай краткий FMECA-анализ отказа подшипников насоса: "
        "режимы отказа, эффекты и профилактика.",
    ),
    (
        "motor_overheat",
        "Сделай краткий FMECA-анализ перегрева электродвигателя, приводящего насос: "
        "причины, последствия и способы предотвращения.",
    ),
    (
        "controller_fault",
        "Сделай краткий FMECA-анализ отказа контроллера/ПИД-регулятора, "
        "управляющего насосом: типовые виды отказов, эффект на систему и детектирование.",
    ),
]


def _one_request(i: int, tokenizer, model) -> None:
    history = init_history()
    scenario_name, question = SCENARIOS[i % len(SCENARIOS)]
    answer = generate(
        tokenizer,
        model,
        history,
        question,
        scenario=scenario_name,  # fixed set of scenario labels
        do_sample=False,
        max_new_tokens=256,
    )
    print(f"[{i}] scenario={scenario_name}, answer_len={len(answer)}")


def main() -> None:
    # Start Prometheus exporter on :8001
    init_prometheus_metrics(port=8001)

    # Warm-up model
    tokenizer, model = load()

    # Warm-up embedder and Qdrant client once to avoid lazy init in threads
    from retrieval import get_embedder, get_qdrant  # local import to avoid cycles
    _ = get_embedder()
    _ = get_qdrant()

    # One burst of parallel requests to demonstrate continuous batching
    n_workers = len(SCENARIOS)
    n_requests = len(SCENARIOS)

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_one_request, i, tokenizer, model)
            for i in range(n_requests)
        ]
        for f in futures:
            f.result()
    total = time.perf_counter() - start
    print(
        f"Completed {n_requests} requests with {n_workers} workers in {total:.2f} s"
    )

if __name__ == "__main__":
    main()