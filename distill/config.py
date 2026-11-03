"""Paths, models, prices, and tiny JSONL helpers."""
from __future__ import annotations
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INPUTS_DIR = DATA / "inputs"
GOLD_DIR = DATA / "gold"
LABELS_DIR = DATA / "teacher_labels"
CURATED_DIR = DATA / "curated"
RUNS_DIR = DATA / "runs"
MODELS_DIR = RUNS_DIR / "models"
BENCH_DIR = RUNS_DIR / "bench"
for _d in (INPUTS_DIR, GOLD_DIR, LABELS_DIR, CURATED_DIR, RUNS_DIR, MODELS_DIR, BENCH_DIR):
    _d.mkdir(parents=True, exist_ok=True)

INPUTS_FILE = INPUTS_DIR / "inputs.jsonl"
POOL_FILE = INPUTS_DIR / "pool.jsonl"          # leakage-safe pool (no gold, no near-dups of gold)
SPLIT_FILE = INPUTS_DIR / "split.json"         # {"train": [ids], "val": [ids]}
GOLD_FILE = GOLD_DIR / "gold.jsonl"
TEACHER_LABELS_FILE = LABELS_DIR / "labels.jsonl"
RATIONALE_FILE = LABELS_DIR / "rationales.jsonl"
FEWSHOT_FILE = DATA / "fewshot.jsonl"          # hand-written examples {"text","label"}; NOT from gold
LABELS_FILE = DATA / "labels.json"


def load_labels() -> list[str]:
    if LABELS_FILE.exists():
        return json.loads(LABELS_FILE.read_text())
    return ["billing", "technical_issue", "account_access", "shipping", "refund", "other"]


LABELS = load_labels()

# ---- Models ----------------------------------------------------------------
TEACHER_MODEL = os.getenv("TEACHER_MODEL", "claude-sonnet-5-5")
MID_MODEL = os.getenv("MID_MODEL", "claude-haiku-5-5")
# Keep this equal to the student's base model so the prompted baseline is a fair comparison.
SMALL_BASELINE_MODEL = os.getenv("SMALL_BASELINE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# USD per 1M tokens (input, output). REPLACE with current numbers from the provider's pricing page.
PRICES = {
    TEACHER_MODEL: (3.0, 15.0),
    MID_MODEL: (1.0, 5.0),
}

GPU_HOURLY_USD = float(os.getenv("GPU_HOURLY_USD", "1.0"))
SEED = 42


# ---- JSONL helpers -----------------------------------------------------------
def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path, row) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")