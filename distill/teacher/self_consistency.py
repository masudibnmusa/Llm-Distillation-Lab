from __future__ import annotations
from collections import Counter
from distill.task.schema import parse_output


def vote(raw_samples: list[str]) -> dict:
    """Majority vote over parsed labels. agreement = votes for winner / total samples."""
    labels = [(parse_output(r).category if parse_output(r) else None) for r in raw_samples]
    valid = [l for l in labels if l]
    if not valid:
        return {"label": None, "agreement": 0.0, "n_valid": 0}
    label, count = Counter(valid).most_common(1)[0]
    return {"label": label, "agreement": count / len(raw_samples), "n_valid": len(valid)}