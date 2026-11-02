"""Is the student's confidence meaningful? Required for a working cascade."""
from __future__ import annotations
import numpy as np
from distill.config import RUNS_DIR, read_jsonl


def reliability(conf, correct, bins: int = 10):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges, out = np.linspace(0, 1, bins + 1), []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            out.append((lo, hi, float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return out


def ece(conf, correct, bins: int = 10) -> float:
    n = len(conf)
    return sum(c / n * abs(acc - mc) for _, _, mc, acc, c in reliability(conf, correct, bins))


def selective_accuracy(conf, correct, thresholds):
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    return [(float(t), float((conf >= t).mean()), float(correct[conf >= t].mean()) if (conf >= t).any() else None)
            for t in thresholds]  # (threshold, coverage, accuracy on covered)


def from_run(run: str) -> dict:
    rows = [r for r in read_jsonl(RUNS_DIR / run / "preds.jsonl") if r["conf"] is not None]
    conf = [r["conf"] for r in rows]
    correct = [int(r["pred"] == r["gold"]) for r in rows]
    return {"ece": ece(conf, correct), "bins": reliability(conf, correct),
            "selective": selective_accuracy(conf, correct, [0.5, 0.7, 0.9, 0.95])}