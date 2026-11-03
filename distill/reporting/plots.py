from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from distill.config import RUNS_DIR

PLOTS = RUNS_DIR / "plots"
PLOTS.mkdir(exist_ok=True)


def pareto(points: list[tuple[str, float, float]], fname: str = "pareto.png"):
    """points = [(name, usd_per_1k, accuracy)]"""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, cost, acc in points:
        ax.scatter(cost, acc); ax.annotate(name, (cost, acc), textcoords="offset points", xytext=(5, 5))
    ax.set_xscale("log"); ax.set_xlabel("USD per 1k requests"); ax.set_ylabel("accuracy"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(PLOTS / fname, dpi=150); plt.close(fig)


def learning_curve(sizes: list[int], accs: list[float], teacher_acc: float | None = None,
                   fname: str = "learning_curve.png"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sizes, accs, marker="o", label="student")
    if teacher_acc:
        ax.axhline(teacher_acc, ls="--", color="gray", label="teacher")
    ax.set_xscale("log"); ax.set_xlabel("training examples"); ax.set_ylabel("gold accuracy")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(PLOTS / fname, dpi=150); plt.close(fig)


def per_class(student: dict, teacher: dict, fname: str = "per_class.png"):
    """student / teacher = metrics['per_class'] dicts"""
    labels = sorted(student, key=lambda k: student[k]["f1-score"])
    fig, ax = plt.subplots(figsize=(7, max(4, len(labels) * 0.25)))
    ax.barh(labels, [student[k]["f1-score"] for k in labels], alpha=.7, label="student")
    ax.scatter([teacher.get(k, {}).get("f1-score", 0) for k in labels], labels, color="k", s=12, label="teacher")
    ax.set_xlabel("F1"); ax.legend()
    fig.tight_layout(); fig.savefig(PLOTS / fname, dpi=150); plt.close(fig)


def reliability_diagram(bins: list[tuple], fname: str = "reliability.png"):
    """bins from calibration.reliability(): (lo, hi, mean_conf, acc, n)"""
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.plot([b[2] for b in bins], [b[3] for b in bins], marker="o")
    ax.set_xlabel("confidence"); ax.set_ylabel("accuracy")
    fig.tight_layout(); fig.savefig(PLOTS / fname, dpi=150); plt.close(fig)