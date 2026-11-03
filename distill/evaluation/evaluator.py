"""Run any system over the gold set; save predictions + metrics under data/runs/<run>/."""
from __future__ import annotations
import json
from distill.config import GOLD_FILE, MID_MODEL, RUNS_DIR, TEACHER_MODEL, read_jsonl, write_jsonl
from distill.evaluation import metrics


def build_system(name: str, model: str | None = None, no_cache: bool = False):
    if name in ("student", "classifier") and not model:
        raise SystemExit(f"--system {name} needs --model <trained tag> (a folder under data/runs/models/)")
    if name in ("teacher", "mid"):
        from distill.baselines.prompted_mid_model import APISystem
        return APISystem(TEACHER_MODEL if name == "teacher" else MID_MODEL, use_cache=not no_cache)
    if name == "prompted_small":
        from distill.baselines.prompted_small_model import PromptedSmall
        return PromptedSmall()
    if name == "tfidf":
        from distill.baselines.classical_baseline import TfidfSystem
        return TfidfSystem().fit_from_curated()
    if name == "student":
        from distill.baselines.prompted_small_model import StudentSystem
        return StudentSystem(model)
    if name == "classifier":
        from distill.student.train_classifier import EncoderSystem
        return EncoderSystem(model)
    raise ValueError(f"unknown system {name}")


def evaluate(system_name: str, run_name: str | None = None, file=None, model: str | None = None) -> dict:
    run = run_name or model or system_name
    system = build_system(system_name, model)
    rows = read_jsonl(file or GOLD_FILE)
    preds = system.predict([r["text"] for r in rows])

    out = RUNS_DIR / run
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "preds.jsonl", [
        {"id": r["id"], "text": r["text"], "gold": r["label"], "pred": p.label, "conf": p.conf,
         "valid": p.valid, "latency_s": p.latency_s, "raw": p.raw} for r, p in zip(rows, preds)])
    m = metrics.compute(preds, [r["label"] for r in rows])
    m["system"] = system_name
    (out / "metrics.json").write_text(json.dumps(m, indent=2))
    return m