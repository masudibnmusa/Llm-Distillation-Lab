"""Bootstrap confidence intervals for accuracy and for paired accuracy differences."""
from __future__ import annotations
import numpy as np
from distill.config import SEED


def bootstrap_ci(correct, n_boot: int = 2000, alpha: float = 0.05, seed: int = SEED) -> tuple[float, float]:
    """95% (by default) percentile CI for the mean of a 0/1 correctness vector."""
    x = np.asarray(correct, float)
    if len(x) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff_ci(a, b, n_boot: int = 2000, alpha: float = 0.05,
                   seed: int = SEED) -> tuple[float, float, float]:
    """mean(a) - mean(b) on the SAME examples, with a paired bootstrap CI.
    Resampling examples (not systems) keeps the pairing, which gives much tighter intervals."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) != len(b):
        raise ValueError("paired_diff_ci needs equal-length, aligned inputs")
    if len(a) == 0:
        return 0.0, 0.0, 0.0
    d = a - b
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
