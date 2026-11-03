"""Track real (non-cached) teacher spend."""
from __future__ import annotations
import threading
from collections import defaultdict
from distill.config import PRICES, RUNS_DIR, append_jsonl, read_jsonl

LEDGER_FILE = RUNS_DIR / "cost_ledger.jsonl"
_lock = threading.Lock()


def usd(model: str, in_tok: int, out_tok: int) -> float:
    pi, po = PRICES.get(model, (0.0, 0.0))
    return (in_tok * pi + out_tok * po) / 1e6


def record(model: str, in_tok: int, out_tok: int, tag: str) -> None:
    with _lock:
        append_jsonl(LEDGER_FILE, {"model": model, "in": in_tok, "out": out_tok, "tag": tag,
                                   "usd": usd(model, in_tok, out_tok)})


def summary() -> dict:
    rows = read_jsonl(LEDGER_FILE)
    by_tag: dict = defaultdict(lambda: {"calls": 0, "usd": 0.0, "in": 0, "out": 0})
    for r in rows:
        t = by_tag[r["tag"]]
        t["calls"] += 1; t["usd"] += r["usd"]; t["in"] += r["in"]; t["out"] += r["out"]
    return {"total_usd": round(sum(r["usd"] for r in rows), 4), "by_tag": dict(by_tag)}


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))