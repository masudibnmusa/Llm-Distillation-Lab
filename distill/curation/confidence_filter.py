from __future__ import annotations


def split_by_agreement(rows: list[dict], min_agreement: float = 0.67):
    """Returns (keep, review). Single-sample runs have agreement 1.0 and are all kept."""
    keep = [r for r in rows if r["agreement"] >= min_agreement]
    review = [r for r in rows if r["agreement"] < min_agreement]
    return keep, review