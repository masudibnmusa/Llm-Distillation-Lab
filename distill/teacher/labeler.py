"""Run the teacher over the pool. Cached, resumable, cost-tracked."""
from __future__ import annotations
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from distill.config import TEACHER_LABELS_FILE, append_jsonl, read_jsonl
from distill.task.prompt_templates import SYSTEM, teacher_messages
from distill.teacher.self_consistency import vote
from distill.teacher.teacher_client import TeacherClient


def label_all(inputs: list[dict], n_samples: int = 1, workers: int = 8,
              model: str | None = None, out=TEACHER_LABELS_FILE) -> None:
    client = TeacherClient(model)
    done = {r["id"] for r in read_jsonl(out)}
    todo = [x for x in inputs if x["id"] not in done]
    lock = threading.Lock()

    def work(x):
        raws = [client.complete(teacher_messages(x["text"]), system=SYSTEM,
                                sample_idx=i, tag="label")[0] for i in range(n_samples)]
        row = {"id": x["id"], "text": x["text"], "raw": raws, **vote(raws)}
        with lock:
            append_jsonl(out, row)

    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(work, todo), total=len(todo), desc="teacher labeling"))