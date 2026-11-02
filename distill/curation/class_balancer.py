from __future__ import annotations
import random
from collections import defaultdict
from statistics import median


def balance(rows: list[dict], max_per_class: int | None = None, seed: int = 42) -> list[dict]:
    """Cap over-represented classes. Default cap = 2x median class size."""
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    if not by:
        return rows
    cap = max_per_class or int(2 * median(len(v) for v in by.values()))
    rng, out = random.Random(seed), []
    for v in by.values():
        rng.shuffle(v)
        out += v[:cap]
    return out