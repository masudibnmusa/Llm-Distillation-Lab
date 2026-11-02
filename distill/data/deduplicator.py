"""Near-duplicate removal (TF-IDF cosine). Leakage between train and gold inflates results."""
from __future__ import annotations
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def near_dup_mask(texts: list[str], ref_texts: list[str], thr: float = 0.9, chunk: int = 1000) -> np.ndarray:
    """True where a text has cosine similarity >= thr to ANY reference text."""
    if not texts or not ref_texts:
        return np.zeros(len(texts), dtype=bool)
    vec = TfidfVectorizer().fit(texts + ref_texts)
    X, R = vec.transform(texts), vec.transform(ref_texts)
    mask = np.zeros(len(texts), dtype=bool)
    for i in range(0, len(texts), chunk):
        sims = (X[i:i + chunk] @ R.T).toarray()
        mask[i:i + chunk] = sims.max(axis=1) >= thr
    return mask


def drop_near_dupes_against(items: list[dict], ref_texts: list[str], thr: float = 0.9) -> list[dict]:
    mask = near_dup_mask([x["text"] for x in items], ref_texts, thr)
    return [x for x, bad in zip(items, mask) if not bad]


def dedupe_within(items: list[dict], thr: float = 0.95, chunk: int = 1000) -> list[dict]:
    """Greedy: drop an item if it is near-identical to any earlier item."""
    if len(items) < 2:
        return items
    X = TfidfVectorizer().fit_transform([x["text"] for x in items])
    dup = np.zeros(len(items), dtype=bool)
    for s in range(0, len(items), chunk):
        S = (X[s:s + chunk] @ X.T).toarray()
        for r in range(S.shape[0]):
            g = s + r
            if g and (S[r, :g] >= thr).any():
                dup[g] = True
    return [x for x, d in zip(items, dup) if not d]