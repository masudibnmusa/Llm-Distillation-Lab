"""Human-labeled evaluation set tooling (terminal)."""
from __future__ import annotations
import random
from distill.config import GOLD_FILE, INPUTS_FILE, LABELS, SEED, TEACHER_MODEL, append_jsonl, read_jsonl


def prompt_label(text: str, default: str | None = None):
    """Returns a label, None (skip) or "QUIT"."""
    print("\n" + "-" * 70 + "\n" + text + "\n")
    hint = f" [Enter = {default}]" if default else ""
    while True:
        s = input(f"label (name prefix or number; l=list, s=skip, q=quit){hint}: ").strip().lower()
        if s == "q":
            return "QUIT"
        if s == "s":
            return None
        if s == "l":
            print("\n".join(f"{i:>3}. {l}" for i, l in enumerate(LABELS)))
        elif s == "" and default:
            return default
        elif s.isdigit() and int(s) < len(LABELS):
            return LABELS[int(s)]
        else:
            m = [l for l in LABELS if l.startswith(s)] if s else []
            if len(m) == 1:
                return m[0]
            print("ambiguous or unknown" if m else "unknown")


def build_gold(n: int = 300, assist: bool = False, use_source_labels: bool = False, seed: int = SEED):
    inputs = read_jsonl(INPUTS_FILE)
    have = {r["id"] for r in read_jsonl(GOLD_FILE)}
    cand = [x for x in inputs if x["id"] not in have]
    random.Random(seed).shuffle(cand)
    need = n - len(have)

    if use_source_labels:
        for x in [c for c in cand if c.get("source_label")][:need]:
            append_jsonl(GOLD_FILE, {"id": x["id"], "text": x["text"], "label": x["source_label"]})
        print(f"gold size: {len(read_jsonl(GOLD_FILE))}")
        return

    teacher = None
    if assist:  # WARNING: anchors you on the teacher's answer and inflates teacher accuracy
        from distill.baselines.prompted_mid_model import APISystem
        teacher = APISystem(TEACHER_MODEL)

    done = len(have)
    for x in cand:
        if done >= n:
            break
        default = teacher.predict_one(x["text"]).label if teacher else None
        lab = prompt_label(x["text"], default)
        if lab == "QUIT":
            break
        if lab:
            append_jsonl(GOLD_FILE, {"id": x["id"], "text": x["text"], "label": lab})
            done += 1
    print(f"gold size: {done}")