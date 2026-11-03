"""distill label | train | eval | report   (+ helper commands)"""
from __future__ import annotations
import argparse
import json


def main():
    p = argparse.ArgumentParser(prog="distill")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("collect", help="gather unlabeled inputs")
    s.add_argument("--hf"); s.add_argument("--hf-config"); s.add_argument("--split", default="train")
    s.add_argument("--text-col", default="text"); s.add_argument("--label-col")
    s.add_argument("--jsonl"); s.add_argument("--csv"); s.add_argument("--limit", type=int)

    s = sub.add_parser("synth", help="teacher-generated inputs to fill gaps")
    s.add_argument("--per-label", type=int, default=50)

    s = sub.add_parser("gold", help="build the gold eval set")
    s.add_argument("--n", type=int, default=300)
    s.add_argument("--assist", action="store_true", help="show teacher suggestion (biases gold!)")
    s.add_argument("--use-source-labels", action="store_true")

    sub.add_parser("split", help="leakage-safe train/val split")

    s = sub.add_parser("label", help="run teacher over the pool")
    s.add_argument("--samples", type=int, default=1); s.add_argument("--workers", type=int, default=8)
    s.add_argument("--model"); s.add_argument("--rationales", action="store_true")

    s = sub.add_parser("curate", help="filter/balance/format training data")
    s.add_argument("--min-agreement", type=float, default=0.67)
    s.add_argument("--max-per-class", type=int); s.add_argument("--n-train", type=int)
    s.add_argument("--rationale", action="store_true")
    s.add_argument("--review", action="store_true", help="interactively review disputed examples")

    s = sub.add_parser("train")
    s.add_argument("--kind", choices=["sft", "kd", "classifier"], default="sft")
    s.add_argument("--student", default="qwen-1.5b")
    s.add_argument("--config", default="distill/student/configs/lora.yaml")
    s.add_argument("--tag"); s.add_argument("--n-train", type=int)
    s.add_argument("--teacher-model", default="Qwen/Qwen2.5-7B-Instruct", help="for --kind kd")

    s = sub.add_parser("eval")
    s.add_argument("--system", required=True,
                   choices=["teacher", "mid", "prompted_small", "tfidf", "student", "classifier"])
    s.add_argument("--model", help="trained model tag (student/classifier)")
    s.add_argument("--run-name"); s.add_argument("--file")

    s = sub.add_parser("robust")
    s.add_argument("--system", required=True); s.add_argument("--model")

    s = sub.add_parser("bench")
    s.add_argument("kind", choices=["latency", "throughput"])
    s.add_argument("--system", default="student"); s.add_argument("--model"); s.add_argument("--run-name")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--url", default="http://localhost:8000/v1/chat/completions")
    s.add_argument("--served-name", default="student")

    s = sub.add_parser("breakeven")
    s.add_argument("--gpu-hourly", type=float); s.add_argument("--student-rps", type=float)
    s.add_argument("--utilization", type=float, default=0.5)
    s.add_argument("--eng-hours", type=float, default=40); s.add_argument("--eng-hourly", type=float, default=0.0)
    s.add_argument("--train-gpu-hours", type=float, default=1.0)
    s.add_argument("--horizon-months", type=int, default=12)

    s = sub.add_parser("cascade")
    s.add_argument("--student-run", required=True); s.add_argument("--teacher-run", default="teacher")

    sub.add_parser("report")
    a = p.parse_args()

    from distill import config as C

    if a.cmd == "collect":
        from distill.data import input_collector as ic
        if a.hf:
            rows = ic.from_hf(a.hf, a.split, a.text_col, a.label_col, a.hf_config, a.limit)
        elif a.jsonl:
            rows = ic.from_jsonl(a.jsonl, a.text_col, a.limit)
        elif a.csv:
            rows = ic.from_csv(a.csv, a.text_col, a.limit)
        else:
            raise SystemExit("give --hf, --jsonl or --csv")
        print(f"saved {ic.save_inputs(rows)} inputs")

    elif a.cmd == "synth":
        from distill.data.synthetic_inputs import generate
        generate(a.per_label)

    elif a.cmd == "gold":
        from distill.data.gold_builder import build_gold
        build_gold(a.n, a.assist, a.use_source_labels)

    elif a.cmd == "split":
        from distill.data.splitter import run_split
        run_split()

    elif a.cmd == "label":
        from distill.teacher.labeler import label_all
        pool = C.read_jsonl(C.POOL_FILE)
        label_all(pool, a.samples, a.workers, a.model)
        if a.rationales:
            from distill.teacher.rationale_generator import generate_rationales
            generate_rationales(a.workers)
        from distill.teacher.cost_ledger import summary
        print(json.dumps(summary(), indent=2))

    elif a.cmd == "curate":
        if a.review:
            from distill.curation.review_queue import review_cli
            review_cli()
        else:
            from distill.curation.format_for_sft import build
            build(a.min_agreement, a.max_per_class, a.rationale, a.n_train)

    elif a.cmd == "train":
        tag = a.tag or a.student
        if a.kind == "sft":
            from distill.student.train_sft import train
            train(a.student, a.config, tag, a.n_train)
        elif a.kind == "kd":
            from distill.student.train_logit_distill import train
            train(a.student, a.teacher_model, tag, a.n_train)
        else:
            from distill.student.train_classifier import train
            train(a.student if a.student != "qwen-1.5b" else "distilbert", tag, a.n_train)

    elif a.cmd == "eval":
        from distill.evaluation.evaluator import evaluate
        m = evaluate(a.system, a.run_name, a.file, a.model)
        print(json.dumps({k: v for k, v in m.items() if k != "per_class"}, indent=2))

    elif a.cmd == "robust":
        from distill.evaluation.evaluator import build_system
        from distill.evaluation.robustness import run
        print(json.dumps(run(build_system(a.system, a.model)), indent=2))

    elif a.cmd == "bench":
        name = a.run_name or a.model or a.system
        if a.kind == "latency":
            from distill.evaluation.evaluator import build_system
            from distill.benchmarking.latency_bench import bench
            bench(build_system(a.system, a.model, no_cache=True), name, a.n)
        else:
            from distill.benchmarking.throughput_bench import run
            run(a.url, a.served_name, name)

    elif a.cmd == "breakeven":
        from distill.benchmarking.breakeven import run
        run(a)

    elif a.cmd == "cascade":
        from distill.serving.cascade_router import sweep
        print(sweep(a.student_run, a.teacher_run).to_string(index=False))

    elif a.cmd == "report":
        from distill.reporting.report_builder import build
        print(f"wrote {build()}")


if __name__ == "__main__":
    main()