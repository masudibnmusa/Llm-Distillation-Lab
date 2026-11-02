from __future__ import annotations
import csv
import hashlib
import json
from distill.config import INPUTS_FILE, LABELS_FILE, read_jsonl, write_jsonl


def from_jsonl(path, text_key="text", limit=None):
    rows = [{"text": r[text_key], "source_label": r.get("label")} for r in read_jsonl(path)]
    return rows[:limit] if limit else rows


def from_csv(path, text_col="text", limit=None):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [{"text": r[text_col], "source_label": r.get("label")} for r in csv.DictReader(f)]
    return rows[:limit] if limit else rows


def from_hf(name, split="train", text_col="text", label_col=None, config=None, limit=None):
    from datasets import load_dataset
    ds = load_dataset(name, config, split=split)
    names = ds.features[label_col].names if label_col and hasattr(ds.features[label_col], "names") else None
    rows = []
    for i, ex in enumerate(ds):
        if limit and i >= limit:
            break
        lab = None
        if label_col:
            v = ex[label_col]
            lab = names[v] if names else str(v)
        rows.append({"text": ex[text_col], "source_label": lab})
    return rows


def _id(text: str) -> str:
    return hashlib.sha1(text.strip().lower().encode()).hexdigest()[:12]


def save_inputs(rows: list[dict], source: str = "real") -> int:
    """Merge into inputs.jsonl, exact-dedupe by normalized text. Writes labels.json if source labels exist."""
    existing = {r["id"]: r for r in read_jsonl(INPUTS_FILE)}
    for r in rows:
        text = (r.get("text") or "").strip()
        if text:
            existing.setdefault(_id(text), {"id": _id(text), "text": text,
                                            "source_label": r.get("source_label"), "source": source})
    write_jsonl(INPUTS_FILE, existing.values())
    labs = sorted({r["source_label"] for r in rows if r.get("source_label")})
    if labs and not LABELS_FILE.exists():
        LABELS_FILE.write_text(json.dumps(labs, indent=2))
        print(f"wrote {len(labs)} labels to {LABELS_FILE}")
    return len(existing)


def load_inputs() -> list[dict]:
    return read_jsonl(INPUTS_FILE)