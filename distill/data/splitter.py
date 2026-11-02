"""Train / val / gold-test split, leakage-safe."""
from __future__ import annotations
import json
import random
from distill.config import (GOLD_FILE, INPUTS_FILE, POOL_FILE, SEED, SPLIT_FILE,
                            read_jsonl, write_jsonl)
from distill.data.deduplicator import drop_near_dupes_against, dedupe_within


def make_split(inputs: list[dict], gold: list[dict], val_frac: float = 0.05,
               seed: int = SEED, thr: float = 0.9):
    gold_ids = {g["id"] for g in gold}
    pool = [x for x in inputs if x["id"] not in gold_ids]
    pool = drop_near_dupes_against(pool, [g["text"] for g in gold], thr)
    pool = dedupe_within(pool, thr=0.95)
    random.Random(seed).shuffle(pool)
    n_val = max(1, int(len(pool) * val_frac))
    split = {"val": [x["id"] for x in pool[:n_val]], "train": [x["id"] for x in pool[n_val:]]}
    return pool, split


def run_split():
    inputs, gold = read_jsonl(INPUTS_FILE), read_jsonl(GOLD_FILE)
    if not gold:
        raise SystemExit("build the gold set first: distill gold")
    pool, split = make_split(inputs, gold)
    write_jsonl(POOL_FILE, pool)
    SPLIT_FILE.write_text(json.dumps(split))
    print(f"inputs={len(inputs)} gold={len(gold)} pool={len(pool)} "
          f"(train={len(split['train'])}, val={len(split['val'])})")