"""Human review for disputed examples."""
from __future__ import annotations
from distill.config import CURATED_DIR, append_jsonl, read_jsonl, write_jsonl
from distill.data.gold_builder import prompt_label

REVIEW_FILE = CURATED_DIR / "review_queue.jsonl"
REVIEWED_FILE = CURATED_DIR / "reviewed.jsonl"


def write_queue(rows: list[dict]) -> None:
    write_jsonl(REVIEW_FILE, [{"id": r["id"], "text": r["text"], "teacher_label": r["label"],
                               "agreement": r["agreement"]} for r in rows])


def review_cli() -> None:
    done = {r["id"] for r in read_jsonl(REVIEWED_FILE)}
    todo = [r for r in read_jsonl(REVIEW_FILE) if r["id"] not in done]
    print(f"{len(todo)} disputed examples to review")
    for r in todo:
        lab = prompt_label(f"{r['text']}\n\n(teacher: {r['teacher_label']}, agreement {r['agreement']:.2f})",
                           default=r["teacher_label"])
        if lab == "QUIT":
            break
        if lab:
            append_jsonl(REVIEWED_FILE, {"id": r["id"], "label": lab})