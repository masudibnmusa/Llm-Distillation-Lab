"""Teacher-generated inputs to fill coverage gaps (hard / rare / messy cases)."""
from __future__ import annotations
import json
import re
from distill.config import LABELS
from distill.data.input_collector import save_inputs
from distill.teacher.teacher_client import TeacherClient


def generate(per_label: int = 50, labels: list[str] | None = None) -> None:
    client = TeacherClient()
    rows = []
    for label in labels or LABELS:
        prompt = (
            f"Write {per_label} diverse, realistic customer support messages that belong to the category "
            f"'{label}'. Vary length, tone, and writing quality (typos, slang, missing context, "
            f"multi-issue messages). Return ONLY a JSON array of strings."
        )
        raw, _ = client.complete([{"role": "user", "content": prompt}], max_tokens=16000, tag="synth")
        m = re.search(r"\[.*\]", raw, re.S)
        if not m:
            print(f"could not parse synthetic batch for {label}")
            continue
        try:
            rows += [{"text": t, "source_label": None} for t in json.loads(m.group(0)) if isinstance(t, str)]
        except Exception:
            print(f"could not parse synthetic batch for {label}")
    print(f"saved, total inputs now {save_inputs(rows, source='synthetic')}")