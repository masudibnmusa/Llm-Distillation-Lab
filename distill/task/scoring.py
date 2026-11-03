from __future__ import annotations
from sklearn.metrics import f1_score
from distill.task.schema import Prediction

INVALID = "__invalid__"


def exact_match(pred, gold) -> int:
    return int(pred is not None and pred == gold)


def accuracy(preds: list, golds: list) -> float:
    return sum(exact_match(p, g) for p, g in zip(preds, golds)) / max(len(golds), 1)


def macro_f1(preds: list, golds: list) -> float:
    return float(f1_score(golds, [p or INVALID for p in preds], average="macro", zero_division=0))


def schema_validity(preds: list[Prediction]) -> float:
    return sum(p.valid for p in preds) / max(len(preds), 1)