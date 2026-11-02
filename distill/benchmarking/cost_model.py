"""$/1k requests: API pricing vs. GPU hours."""
from __future__ import annotations
from distill.config import TEACHER_MODEL
from distill.teacher.cost_ledger import LEDGER_FILE
from distill.config import read_jsonl


def teacher_usd_per_request(model: str = TEACHER_MODEL) -> float:
    """Average real cost per single teacher call, measured from the ledger (prefers eval-time calls)."""
    rows = [r for r in read_jsonl(LEDGER_FILE) if r["model"] == model]
    pick = [r for r in rows if r["tag"] == "eval"] or [r for r in rows if r["tag"] == "label"]
    if not pick:
        raise SystemExit("no ledger data; run `distill eval --system teacher` first")
    return sum(r["usd"] for r in pick) / len(pick)


def gpu_usd_per_1k(gpu_hourly_usd: float, rps: float, utilization: float = 0.5) -> float:
    """Cost per 1k requests if the GPU is paid by the hour and runs at `utilization` of peak rps."""
    return gpu_hourly_usd / (rps * utilization * 3600) * 1000


def api_usd_per_1k(usd_per_request: float) -> float:
    return usd_per_request * 1000