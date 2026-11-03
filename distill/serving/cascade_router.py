"""Student first, escalate to the teacher when the student is unsure."""
from __future__ import annotations
import numpy as np
import pandas as pd
import json
from distill.benchmarking.cost_model import gpu_usd_per_1k, teacher_usd_per_request
from distill.config import BENCH_DIR, GPU_HOURLY_USD, RUNS_DIR, read_jsonl
from distill.task.schema import Prediction


class CascadeSystem:
    def __init__(self, student, teacher, threshold: float = 0.8):
        self.student, self.teacher, self.threshold = student, teacher, threshold
        self.name = f"cascade@{threshold}"

    def predict(self, texts: list[str]) -> list[Prediction]:
        preds = self.student.predict(texts)
        route = [i for i, p in enumerate(preds) if p.conf is None or p.conf < self.threshold or not p.valid]
        for i, tp in zip(route, self.teacher.predict([texts[i] for i in route])):
            preds[i] = tp
        return preds


def _measured_student_usd_per_1k(utilization: float = 0.5) -> float:
    files = sorted(BENCH_DIR.glob("throughput_*.json"))
    if not files:
        print("warning: no throughput bench found; student serving cost counted as $0 "
              "(pass --student-usd-per-1k or run `bench throughput` first)")
        return 0.0
    rps = json.loads(files[-1].read_text())["max_rps"]
    return gpu_usd_per_1k(GPU_HOURLY_USD, rps, utilization)


def sweep(student_run: str, teacher_run: str = "teacher", thresholds=None,
          student_usd_per_1k: float | None = None) -> pd.DataFrame:
    """Offline sweep on saved gold predictions: quality vs. fraction routed to the teacher (no new API calls).
    Every request pays the student cost; routed requests also pay the teacher."""
    if student_usd_per_1k is None:
        student_usd_per_1k = _measured_student_usd_per_1k()
    s = {r["id"]: r for r in read_jsonl(RUNS_DIR / student_run / "preds.jsonl")}
    t = {r["id"]: r for r in read_jsonl(RUNS_DIR / teacher_run / "preds.jsonl")}
    ids = [i for i in s if i in t]
    t1k = teacher_usd_per_request() * 1000
    rows = []
    for thr in (thresholds if thresholds is not None else np.linspace(0, 1, 21)):
        routed = [(s[i]["conf"] is None) or (s[i]["conf"] < thr) or not s[i]["valid"] for i in ids]
        correct = [(t[i] if r else s[i])["pred"] == s[i]["gold"] for i, r in zip(ids, routed)]
        frac = float(np.mean(routed))
        rows.append({"threshold": round(float(thr), 2), "teacher_fraction": frac,
                     "accuracy": float(np.mean(correct)), "usd_per_1k": student_usd_per_1k + frac * t1k})
    df = pd.DataFrame(rows)
    df.to_csv(BENCH_DIR / "cascade.csv", index=False)
    return df