from __future__ import annotations
from sklearn.metrics import classification_report
from distill.evaluation.significance import bootstrap_ci
from distill.task.scoring import INVALID, accuracy, exact_match, macro_f1, schema_validity


def compute(preds, golds) -> dict:
    correct = [exact_match(p.label, g) for p, g in zip(preds, golds)]
    lo, hi = bootstrap_ci(correct)
    rep = classification_report(golds, [p.label or INVALID for p in preds], labels=sorted(set(golds)),
                                output_dict=True, zero_division=0)
    return {"n": len(golds),
            "accuracy": accuracy([p.label for p in preds], golds),
            "accuracy_ci": [lo, hi],
            "macro_f1": macro_f1([p.label for p in preds], golds),
            "schema_validity": schema_validity(preds),
            "per_class": {k: v for k, v in rep.items() if k in set(golds)}}