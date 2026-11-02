"""Student vs. teacher match rate (not the same as accuracy!)."""
from __future__ import annotations
from distill.config import RUNS_DIR, read_jsonl


def agreement(run: str, teacher_run: str = "teacher") -> dict:
    s = {r["id"]: r for r in read_jsonl(RUNS_DIR / run / "preds.jsonl")}
    t = {r["id"]: r for r in read_jsonl(RUNS_DIR / teacher_run / "preds.jsonl")}
    ids = [i for i in s if i in t]
    same = [s[i]["pred"] == t[i]["pred"] for i in ids]
    t_right = [t[i]["pred"] == t[i]["gold"] for i in ids]
    def rate(mask):
        sel = [x for x, m in zip(same, mask) if m]
        return sum(sel) / len(sel) if sel else None
    return {"n": len(ids), "agreement": sum(same) / max(len(ids), 1),
            "agreement_when_teacher_right": rate(t_right),
            "agreement_when_teacher_wrong": rate([not x for x in t_right])}  # high = copying teacher errors