"""Requests/sec under concurrency against a vLLM OpenAI-compatible server (`distill.serving.serve`)."""
from __future__ import annotations
import json
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
from distill.config import BENCH_DIR, GOLD_FILE, read_jsonl
from distill.task.prompt_templates import student_messages


def _payload(model, text, stream=False):
    return {"model": model, "messages": student_messages(text), "max_tokens": 32,
            "temperature": 0, "stream": stream}


def _one(url, model, text):
    t0 = time.perf_counter()
    requests.post(url, json=_payload(model, text), timeout=120).raise_for_status()
    return time.perf_counter() - t0


def ttft(url, model, text) -> float:
    t0 = time.perf_counter()
    with requests.post(url, json=_payload(model, text, True), stream=True, timeout=120) as r:
        for line in r.iter_lines():
            if line.startswith(b"data:") and b"[DONE]" not in line:
                delta = json.loads(line[5:])["choices"][0].get("delta", {})
                if delta.get("content"):
                    return time.perf_counter() - t0
    return float("nan")


def run(url: str, model: str, name: str, concurrencies=(1, 4, 16, 64), n: int = 200) -> list[dict]:
    texts = [r["text"] for r in read_jsonl(GOLD_FILE)]
    reqs = (texts * (n // max(len(texts), 1) + 1))[:n]
    _one(url, model, reqs[0])  # warmup
    out = []
    for c in concurrencies:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(c) as ex:
            lats = list(ex.map(lambda t: _one(url, model, t), reqs))
        wall = time.perf_counter() - t0
        out.append({"concurrency": c, "rps": n / wall, "p50_s": float(np.percentile(lats, 50)),
                    "p95_s": float(np.percentile(lats, 95))})
        print(out[-1])
    res = {"system": name, "ttft_s": ttft(url, model, reqs[0]), "levels": out,
           "max_rps": max(o["rps"] for o in out)}
    (BENCH_DIR / f"throughput_{name}.json").write_text(json.dumps(res, indent=2))
    return out