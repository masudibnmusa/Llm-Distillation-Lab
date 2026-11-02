"""Volume where the student beats the teacher on total cost, plus sensitivity."""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from distill.benchmarking.cost_model import gpu_usd_per_1k, teacher_usd_per_request
from distill.config import BENCH_DIR, GPU_HOURLY_USD, MODELS_DIR
from distill.teacher.cost_ledger import summary


@dataclass
class Assumptions:
    teacher_usd_per_req: float
    labeling_usd: float            # one-time teacher spend
    train_gpu_hours: float
    gpu_hourly_usd: float
    student_rps: float             # peak sustained rps of ONE GPU (from throughput bench)
    utilization: float = 0.5       # fraction of peak you can realistically use
    eng_hours: float = 40
    eng_hourly_usd: float = 0.0    # set > 0 to price your own time
    monitoring_usd_month: float = 0.0
    retrain_usd_year: float = 0.0
    horizon_months: int = 12
    hours_per_month: float = 730


def one_time_cost(a: Assumptions) -> float:
    return a.labeling_usd + a.train_gpu_hours * a.gpu_hourly_usd + a.eng_hours * a.eng_hourly_usd


def teacher_monthly(a: Assumptions, volume: float) -> float:
    return volume * a.teacher_usd_per_req


def student_monthly(a: Assumptions, volume: float) -> float:
    capacity = a.student_rps * a.utilization * 3600 * a.hours_per_month
    gpus = max(1, math.ceil(volume / capacity))
    return (gpus * a.gpu_hourly_usd * a.hours_per_month + a.monitoring_usd_month
            + a.retrain_usd_year / 12 + one_time_cost(a) / a.horizon_months)


def breakeven_volume(a: Assumptions, lo: float = 1e3, hi: float = 1e10, steps: int = 3000):
    """Requests/month at which the student's amortized cost <= teacher cost. None = never in range."""
    for v in np.geomspace(lo, hi, steps):
        if student_monthly(a, v) <= teacher_monthly(a, v):
            return float(v)
    return None


def sensitivity(a: Assumptions, gpu_hourly=(0.5, 1, 2, 4), utilization=(0.25, 0.5, 0.75)) -> pd.DataFrame:
    rows = [{"gpu_hourly_usd": g, **{f"util={u}": breakeven_volume(replace(a, gpu_hourly_usd=g, utilization=u))
                                     for u in utilization}} for g in gpu_hourly]
    return pd.DataFrame(rows)


def run(args) -> dict:
    """CLI entry: build assumptions from measured data + flags, print and save."""
    led = summary()["by_tag"]
    labeling = sum(v["usd"] for k, v in led.items() if k != "eval")
    rps = args.student_rps
    if rps is None:
        files = sorted(BENCH_DIR.glob("throughput_*.json"))
        rps = json.loads(files[-1].read_text())["max_rps"] if files else 10.0
    gpu = args.gpu_hourly or GPU_HOURLY_USD
    a = Assumptions(teacher_usd_per_req=teacher_usd_per_request(), labeling_usd=labeling,
                    train_gpu_hours=args.train_gpu_hours, gpu_hourly_usd=gpu, student_rps=rps,
                    utilization=args.utilization, eng_hours=args.eng_hours, eng_hourly_usd=args.eng_hourly,
                    horizon_months=args.horizon_months)
    be = breakeven_volume(a)
    res = {"assumptions": a.__dict__, "one_time_usd": one_time_cost(a),
           "teacher_usd_per_1k": a.teacher_usd_per_req * 1000,
           "student_gpu_usd_per_1k_at_util": gpu_usd_per_1k(gpu, rps, a.utilization),
           "breakeven_requests_per_month": be, "sensitivity": sensitivity(a).to_dict("records")}
    (BENCH_DIR / "breakeven.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "sensitivity"}, indent=2))
    print(sensitivity(a).to_string(index=False))
    return res