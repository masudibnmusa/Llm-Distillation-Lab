from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel, field_validator
from distill.config import LABELS


class TaskOutput(BaseModel):
    category: str

    @field_validator("category")
    @classmethod
    def _in_labels(cls, v: str) -> str:
        if v not in LABELS:
            raise ValueError(f"unknown label: {v}")
        return v


@dataclass
class Prediction:
    label: Optional[str]
    conf: Optional[float] = None
    valid: bool = True
    latency_s: float = 0.0
    raw: str = ""


_FENCE = re.compile(r"```(?:json)?")


def parse_output(text: str) -> Optional[TaskOutput]:
    """Return a validated TaskOutput, or None if malformed / unknown label."""
    try:
        cleaned = _FENCE.sub("", text).strip()
        m = re.search(r"\{.*\}", cleaned, re.S)
        return TaskOutput(**json.loads(m.group(0))) if m else None
    except Exception:
        return None


def to_json(label: str) -> str:
    return json.dumps({"category": label})


def to_prediction(raw: str, conf=None, latency_s: float = 0.0) -> Prediction:
    out = parse_output(raw)
    return Prediction(out.category if out else None, conf, out is not None, latency_s, raw)