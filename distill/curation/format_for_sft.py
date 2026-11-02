"""Curate teacher labels and write train/val JSONL in chat prompt/completion format."""
from __future__ import annotations
import json
import random
from distill.config import (CURATED_DIR, RATIONALE_FILE, SEED, SPLIT_FILE, TEACHER_LABELS_FILE,
                            read_jsonl, write_jsonl)
from distill.curation.class_balancer import balance
from distill.curation.confidence_filter import split_by_agreement
from distill.curation.review_queue import REVIEWED_FILE, write_queue
from distill.curation.validity_filter import filter_valid
from distill.task.prompt_templates import student_messages
from distill.task.schema import to_json


def to_example(text: str, label: str, rationale: str | None = None) -> dict:
    completion = (rationale + "\n" if rationale else "") + to_json(label)
    return {"prompt": student_messages(text),
            "completion": [{"role": "assistant", "content": completion}],
            "text": text, "label": label}


def build(min_agreement: float = 0.67, max_per_class: int | None = None, rationale: bool = False,
          n_train: int | None = None, seed: int = SEED) -> None:
    rows = {r["id"]: r for r in read_jsonl(TEACHER_LABELS_FILE)}
    split = json.loads(SPLIT_FILE.read_text())
    reviewed = {r["id"]: r["label"] for r in read_jsonl(REVIEWED_FILE)}
    rats = {r["id"]: r["rationale"] for r in read_jsonl(RATIONALE_FILE)} if rationale else {}

    def collect(ids):
        cand = filter_valid([rows[i] for i in ids if i in rows])
        keep, review = split_by_agreement(cand, min_agreement)
        keep += [{**r, "label": reviewed[r["id"]]} for r in review if r["id"] in reviewed]
        return keep, [r for r in review if r["id"] not in reviewed]

    train, queue = collect(split["train"])
    val, _ = collect(split["val"])
    write_queue(queue)
    train = balance(train, max_per_class, seed)
    random.Random(seed).shuffle(train)
    if n_train:
        train = train[:n_train]
    if rationale:
        train = [r for r in train if r["id"] in rats]

    mk = lambda r: to_example(r["text"], r["label"], rats.get(r["id"]))
    name = f"train_{n_train}.jsonl" if n_train else "train.jsonl"
    write_jsonl(CURATED_DIR / name, [mk(r) for r in train])
    write_jsonl(CURATED_DIR / "val.jsonl", [mk(r) for r in val])
    print(f"{name}: {len(train)} train | val: {len(val)} | sent to review: {len(queue)}")