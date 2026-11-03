"""Quality vs. cost vs. latency summary -> data/runs/report.md"""
from __future__ import annotations
import json
from distill.config import BENCH_DIR, RUNS_DIR, read_jsonl
from distill.evaluation import calibration
from distill.evaluation.significance import paired_diff_ci


def _table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _correct(run: str) -> dict:
    return {r["id"]: int(r["pred"] == r["gold"]) for r in read_jsonl(RUNS_DIR / run / "preds.jsonl")}


def _load(path):
    return json.loads(path.read_text()) if path.exists() else None


def build() -> str:
    systems = {d.name: json.loads((d / "metrics.json").read_text())
               for d in sorted(RUNS_DIR.iterdir()) if d.is_dir() and (d / "metrics.json").exists()}
    teacher = _correct("teacher") if "teacher" in systems else None
    L = ["# Distillation report", ""]

    rows = []
    for name, m in systems.items():
        lo, hi = m["accuracy_ci"]
        delta = "-"
        if teacher and name != "teacher":
            c = _correct(name); ids = [i for i in c if i in teacher]
            d, dlo, dhi = paired_diff_ci([c[i] for i in ids], [teacher[i] for i in ids])
            delta = f"{d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
        lat = _load(BENCH_DIR / f"latency_{name}.json")
        rows.append([name, f"{m['accuracy']:.3f} [{lo:.3f}, {hi:.3f}]", f"{m['macro_f1']:.3f}",
                     f"{m['schema_validity']:.3f}", delta,
                     f"{lat['p50_s']*1000:.0f} ms" if lat else "-", f"{lat['p95_s']*1000:.0f} ms" if lat else "-"])
    L += ["## Quality and latency", "",
          _table(["system", "accuracy [95% CI]", "macro-F1", "schema valid", "Δ vs teacher [95% CI]", "p50", "p95"], rows),
          "", "A Δ interval that includes 0 means the gap to the teacher is not statistically distinguishable.", ""]

    lc = sorted((n for n in systems if n.startswith("lc_")), key=lambda n: int(n.split("_")[1]))
    if lc:
        L += ["## Learning curve", "", _table(["train size", "accuracy"],
                                              [[n.split("_")[1], f"{systems[n]['accuracy']:.3f}"] for n in lc]), ""]

    be = _load(BENCH_DIR / "breakeven.json")
    if be:
        v = be["breakeven_requests_per_month"]
        L += ["## Cost and break-even", "",
              f"- Teacher: ${be['teacher_usd_per_1k']:.3f} per 1k requests",
              f"- Student GPU cost at assumed utilization: ${be['student_gpu_usd_per_1k_at_util']:.4f} per 1k requests",
              f"- One-time cost (labeling + training + time): ${be['one_time_usd']:.2f}",
              f"- **Break-even: {'%.0f requests/month' % v if v else 'not reached in the tested range'}**", ""]
        if be.get("sensitivity"):
            L += [_table(list(be["sensitivity"][0].keys()),
                         [[x if k == "gpu_hourly_usd" else ("never" if x is None else "%.0f" % x)
                           for k, x in r.items()] for r in be["sensitivity"]]), ""]

    casc = BENCH_DIR / "cascade.csv"
    if casc.exists():
        import pandas as pd
        df = pd.read_csv(casc)
        L += ["## Cascade (student -> teacher)", "",
              _table(list(df.columns), df.iloc[::4].round(3).values.tolist()), ""]

    for name, m in systems.items():
        if name not in ("teacher", "mid", "tfidf") and any(r["conf"] is not None for r in
                                                           read_jsonl(RUNS_DIR / name / "preds.jsonl")[:5]):
            L += [f"## Calibration: {name}", "", f"ECE = {calibration.from_run(name)['ece']:.3f}", ""]

    path = RUNS_DIR / "report.md"
    path.write_text("\n".join(L))
    return str(path)