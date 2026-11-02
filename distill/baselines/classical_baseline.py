"""TF-IDF + logistic regression: a sanity floor, trained on the same teacher labels as the student."""
from __future__ import annotations
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from distill.config import CURATED_DIR, read_jsonl
from distill.task.schema import Prediction


class TfidfSystem:
    name = "tfidf"

    def __init__(self):
        self.pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                                  LogisticRegression(max_iter=1000, C=10))

    def fit_from_curated(self, path=None):
        rows = read_jsonl(path or CURATED_DIR / "train.jsonl")
        self.pipe.fit([r["text"] for r in rows], [r["label"] for r in rows])
        return self

    def predict(self, texts: list[str]) -> list[Prediction]:
        t0 = time.perf_counter()
        probs = self.pipe.predict_proba(texts)
        dt = (time.perf_counter() - t0) / max(len(texts), 1)
        classes = self.pipe.classes_
        return [Prediction(classes[p.argmax()], float(p.max()), True, dt, classes[p.argmax()]) for p in probs]