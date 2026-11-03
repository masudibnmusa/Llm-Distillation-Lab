from __future__ import annotations
import json
from distill.config import LABELS, FEWSHOT_FILE, read_jsonl

SYSTEM = (
    "You are a support-ticket classifier.\n"
    "Classify the ticket into exactly one of these categories:\n"
    + "\n".join(f"- {l}" for l in LABELS)
    + '\nReply with JSON only, no explanation: {"category": "<one category>"}'
)

# The student learns the label set from training data, so its prompt stays short (fewer tokens = cheaper).
STUDENT_SYSTEM = 'Classify the support ticket. Reply with JSON only: {"category": "<category>"}'


def teacher_messages(text: str, use_fewshot: bool = True) -> list[dict]:
    msgs: list[dict] = []
    if use_fewshot:
        for ex in read_jsonl(FEWSHOT_FILE):
            msgs.append({"role": "user", "content": ex["text"]})
            msgs.append({"role": "assistant", "content": json.dumps({"category": ex["label"]})})
    msgs.append({"role": "user", "content": text})
    return msgs


def full_messages(text: str, use_fewshot: bool = True) -> list[dict]:
    """System + few-shot + input, for local chat models (prompted baseline)."""
    return [{"role": "system", "content": SYSTEM}] + teacher_messages(text, use_fewshot)


def student_messages(text: str) -> list[dict]:
    return [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": text}]