from __future__ import annotations
import sys
from collections import Counter
import numpy as np
from distill.config import CURATED_DIR, read_jsonl


def class_balance(rows: list[dict], key: str = "label") -> dict:
    return dict(Counter(r[key] for r in rows).most_common())


def length_stats(rows: list[dict]) -> dict:
    n = np.array([len(r["text"].split()) for r in rows])
    return {"n": len(n), "mean": float(n.mean()), "p50": float(np.median(n)),
            "p95": float(np.percentile(n, 95)), "max": int(n.max())}


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CURATED_DIR / "train.jsonl"
    rows = read_jsonl(path)
    print(length_stats(rows))
    print(class_balance(rows))