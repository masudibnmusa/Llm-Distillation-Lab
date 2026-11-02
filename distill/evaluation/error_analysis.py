"""Where does the student fail, grouped by cause?"""
from __future__ import annotations
from collections import Counter
import pandas as pd
from distill.config import RUNS_DIR, read_jsonl


def analyze(run: str, teacher_run: str = "teacher") -> dict:
    rows = read_jsonl(RUNS_DIR / run / "preds.jsonl")
    teacher = {r["id"]: r for r in read_jsonl(RUNS_DIR / teacher_run / "preds.jsonl")}
    errs = [r for r in rows if r["pred"] != r["gold"]]
    pairs = Counter((r["gold"], r["pred"]) for r in errs)

    def cause(r):
        if not r["valid"]:
            return "invalid_output"
        t = teacher.get(r["id"])
        if t and t["pred"] == r["pred"]:
            return "inherited_teacher_error"
        if r.get("conf") is not None and r["conf"] < 0.5:
            return "low_confidence"
        if len(r["text"].split()) < 4:
            return "very_short_input"
        if pairs[(r["gold"], r["pred"])] >= 3:
            return "systematic_confusion"
        return "other"

    df = pd.DataFrame([{**r, "cause": cause(r)} for r in errs])
    if len(df):
        df.drop(columns=["raw"], errors="ignore").to_csv(RUNS_DIR / run / "errors.csv", index=False)
    per_class = Counter(r["gold"] for r in rows)
    wrong = Counter(r["gold"] for r in errs)
    return {"n_errors": len(errs),
            "by_cause": dict(Counter(df["cause"])) if len(df) else {},
            "top_confusions": pairs.most_common(10),
            "worst_classes": sorted(((c, wrong[c] / per_class[c], per_class[c]) for c in per_class),
                                    key=lambda x: -x[1])[:10]}