"""Optional: short reasoning traces for rationale distillation (given the voted label)."""
from __future__ import annotations
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from distill.config import RATIONALE_FILE, TEACHER_LABELS_FILE, append_jsonl, read_jsonl
from distill.teacher.teacher_client import TeacherClient


def generate_rationales(workers: int = 8, min_agreement: float = 0.67) -> None:
    client = TeacherClient()
    rows = [r for r in read_jsonl(TEACHER_LABELS_FILE) if r["label"] and r["agreement"] >= min_agreement]
    done = {r["id"] for r in read_jsonl(RATIONALE_FILE)}
    todo = [r for r in rows if r["id"] not in done]
    lock = threading.Lock()

    def work(r):
        prompt = (f"Support message:\n{r['text']}\n\nCorrect category: {r['label']}\n"
                  "In one or two sentences, explain why this message belongs to that category. "
                  "Be concrete and do not mention that you were given the answer.")
        text, _ = client.complete([{"role": "user", "content": prompt}], max_tokens=1024, tag="rationale")
        with lock:
            append_jsonl(RATIONALE_FILE, {"id": r["id"], "rationale": text.strip()})

    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(work, todo), total=len(todo), desc="rationales"))