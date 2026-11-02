"""End-to-end per-request latency, sequential (batch size 1)."""
from __future__ import annotations
import json
import time
import numpy as np
from distill.config import BENCH_DIR, GOLD_FILE, read_jsonl


def bench(system, name: str, n: int = 100, warmup: int = 3) -> dict:
    texts = [r["text"] for r in read_jsonl(GOLD_FILE)][: n + warmup]
    for t in texts[:warmup]:
        system.predict([t])
    lats = []
    for t in texts[warmup:]:
        t0 = time.perf_counter()
        system.predict([t])
        lats.append(time.perf_counter() - t0)
    res = {"system": name, "n": len(lats), "p50_s": float(np.percentile(lats, 50)),
           "p95_s": float(np.percentile(lats, 95)), "mean_s": float(np.mean(lats))}
    (BENCH_DIR / f"latency_{name}.json").write_text(json.dumps(res, indent=2))
    print(res)
    return res