# distillation-lab

Distill a large model's behavior on a **narrow task** into a **small, fast, cheap model**, then measure the tradeoff honestly.

The deliverable is not just a trained model. It is a measured answer to three questions:

1. **Quality:** how much of the teacher's accuracy does the student keep?
2. **Cost and latency:** how much cheaper and faster is it per 1k requests?
3. **Break-even:** at what request volume does the student pay for itself, and where does it fail?

If the answer turns out to be "it doesn't pay off," that is a valid result and the report should say so.

---

## How it works

```
Unlabeled task inputs
        ↓
teacher/labeler.py (+ self_consistency.py)   ← cached, cost-tracked
        ↓
curation/ (validity, confidence, balance)
        ↓
format_for_sft.py → data/curated/train.jsonl
        ↓
student/train_sft.py (LoRA / full FT)
        ↓
evaluation/evaluator.py on the gold set
   ├── teacher                 (ceiling)
   ├── student (distilled)
   ├── prompted small model    (baseline to beat)
   └── TF-IDF classifier       (sanity floor)
        ↓
benchmarking/ (latency, throughput, cost, break-even)
        ↓
report_builder.py → quality vs. cost vs. latency tradeoff
        ↓
errors → more teacher labels on weak areas → retrain
```

**Core idea:** use the teacher to label a large pool of unlabeled inputs, fine-tune the student on the resulting `(input, teacher output)` pairs, then evaluate the student against both the teacher and human-verified ground truth on a held-out set.

---

## Task definition

Pick **one** narrow, measurable task. The narrower the task, the better distillation works.

| Field | Value |
|---|---|
| Task | _e.g. classify support tickets into N categories_ |
| Input format | _e.g. free-text ticket, 1-500 tokens_ |
| Output format | _e.g. JSON `{"category": "...", "confidence": 0.0}`_ |
| Dataset | _e.g. Banking77, or your own inputs_ |
| Success criteria | _e.g. student within 2 points of teacher accuracy on gold set_ |
| Out of scope | The student is **not** a general assistant. Off-task inputs should be rejected. |

Good tasks: ticket classification, invoice field extraction, query routing, fixed-format summarization.
Bad tasks: "be a general assistant."

Full spec lives in `distill/task/task_spec.md`.

---

## Project structure

```
distillation-lab/
├── distill/
│   ├── cli.py                  # distill label | train | eval | report
│   ├── config.py               # teacher/student models, task spec, paths
│   ├── task/                   # spec, Pydantic schema, teacher prompts, scoring
│   ├── data/                   # input collection, gold builder, splitter, dedup, stats
│   ├── teacher/                # API client, labeler, self-consistency, cost ledger
│   ├── curation/               # validity/confidence filters, balancing, review queue, SFT formatting
│   ├── student/                # SFT (LoRA/full), optional logit distill, optional encoder classifier
│   ├── baselines/              # prompted small/mid model, TF-IDF + logistic regression
│   ├── evaluation/             # evaluator, metrics, agreement, error analysis, calibration, significance, robustness
│   ├── benchmarking/           # latency, throughput, cost model, break-even
│   ├── serving/                # quantize, serve (vLLM/Ollama), cascade router
│   └── reporting/              # report builder, plots
├── data/
│   ├── inputs/                 # unlabeled task inputs
│   ├── gold/                   # human-verified eval set (NEVER trained on)
│   ├── teacher_labels/         # cached teacher outputs (JSONL)
│   ├── curated/                # train.jsonl / val.jsonl
│   └── runs/                   # results per experiment
├── notebooks/                  # label inspection, error analysis
├── tests/                      # validity filter, splitter leakage, scoring, break-even
├── .env.example
├── requirements.txt
├── README.md
└── run.sh
```

---

## Setup

**Requirements**

- Python 3.10+
- A GPU for student fine-tuning (see [Compute](#compute))
- An API key for the teacher model (or an open teacher you can host)

**Install**

```bash
git clone https://github.com/masudibnmusa/Llm-Distillation-Lab.git
cd distillation-lab

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env             # then add your API keys
```

**`requirements.txt` (core)**

```
transformers
trl
peft
datasets
vllm
anthropic
scikit-learn
pydantic
pandas
matplotlib
```

**`.env.example`**

```
ANTHROPIC_API_KEY=your-key-here
HF_TOKEN=your-huggingface-token
```

### Compute

| Student size | Typical setup |
|---|---|
| 0.5B-1.5B | Free/low-cost Colab GPU, LoRA |
| 3B | 16-24 GB GPU, LoRA/QLoRA |
| 8B | 24 GB+ GPU, QLoRA |

Decide early whether you are using Colab, a rented GPU, or local hardware. If compute is tight, start with 0.5B-1.5B students.

---

## Quickstart (v1 path)

Keep the first version small. Add the rest only after this works end to end.

**v1 scope:** schema, gold set, teacher labeling, one student fine-tune, evaluation, break-even.

```bash
# 1. Define the task: edit distill/task/task_spec.md, schema.py, prompt_templates.py

# 2. Collect inputs and build the gold set (200 examples to start)
python -m distill.cli collect --hf mteb/banking77 --text-col text --label-col label_text --limit 10000
python -m distill.cli gold --n 200            # label by hand; or add --use-source-labels

# 3. Split with leakage protection
python -m distill.cli split

# 4. Measure the teacher ceiling on gold
python -m distill.cli eval --system teacher

# 5. Run baselines
python -m distill.cli eval --system prompted_small

# 6. Label the input pool (cached and cost-tracked), then curate
python -m distill.cli label --samples 3
python -m distill.cli curate --min-agreement 0.67
python -m distill.cli eval --system tfidf     # trained on the curated teacher labels

# 7. Train one student
python -m distill.cli train --student qwen-1.5b --config distill/student/configs/lora.yaml --tag qwen-1.5b

# 8. Evaluate the student and compare
python -m distill.cli eval --system student --model qwen-1.5b

# 9. Break-even analysis and report
python -m distill.cli breakeven --gpu-hourly 1.0 --student-rps 20
python -m distill.cli report
```

`run.sh` runs the full pipeline, including the learning curve, robustness, cascade and latency benchmarks.

---

## CLI reference

All commands run as `python -m distill.cli <command>`. Every evaluation uses the gold set unless you pass `--file`.

| Command | Purpose |
|---|---|
| `collect` | Gather unlabeled inputs from a HF dataset, JSONL or CSV |
| `synth` | Teacher-generated inputs to fill coverage gaps |
| `gold` | Build the human-labeled gold eval set |
| `split` | Leakage-safe train/val split (drops gold and near-duplicates of gold) |
| `label` | Run the teacher over the pool, with caching, retries, optional self-consistency voting |
| `curate` | Filter, balance and format training data (`--review` to hand-label disputed examples) |
| `train` | Fine-tune the student (`--kind sft` LoRA/full, `kd` logit distillation, `classifier` encoder) |
| `eval` | Evaluate any system (teacher, mid, prompted_small, tfidf, student, classifier) |
| `robust` | Accuracy under perturbations, and confidence on off-task inputs |
| `bench` | `latency` (sequential) or `throughput` (against a vLLM server) |
| `breakeven` | Request volume at which the student pays for itself, with sensitivity table |
| `cascade` | Offline sweep of student-then-teacher routing thresholds |
| `report` | Build the quality vs. cost vs. latency summary |

---

## Systems compared

| System | Role |
|---|---|
| **Teacher** (large API/open model) | Quality ceiling, cost reference |
| **Student** (distilled small model) | The thing being evaluated |
| **Prompted small model** | Baseline to beat: same size as student, few-shot prompt, no training |
| **Prompted mid-tier model** | Cheaper tier of the teacher family |
| **TF-IDF + logistic regression** | Sanity floor for classification tasks |
| **Cascade** (student, then teacher) | Student handles confident inputs, teacher handles the rest |

If the prompted small model gets close to the student, distillation may not be worth it. Always include that baseline.

---

## Evaluation methodology

- **Gold set:** 200-500 human-verified examples, held out and never used for training.
- **Measure against truth, not only the teacher.** Student-teacher agreement alone hides copied noise.
- **Metrics:** accuracy, F1, schema validity rate, judge score where applicable.
- **Breakdowns:** per category and per difficulty.
- **Significance:** bootstrap confidence intervals on the quality gap.
- **Leakage control:** deduplicate near-duplicates across train/val/gold.
- **Robustness:** out-of-distribution, messy, and adversarial inputs tested explicitly.
- **Calibration:** check that the student's confidence is meaningful (required for cascades).
- **Learning curve:** train on 500, 1k, 2k, 5k, 10k examples to find where quality plateaus.

---

## Cost, latency, and break-even

For each system, measure:

- **Latency:** p50 / p95, time to first token, tokens/sec
- **Throughput:** requests/sec under realistic concurrency (not a single request)
- **Cost per 1k requests:** API pricing for the teacher, GPU hours for the student

**Break-even** is the request volume where the one-time distillation cost plus student hosting beats paying the teacher per call.

Include in the accounting:

- Teacher labeling spend
- GPU training cost
- Student hosting and monitoring
- Retraining when the task drifts
- Engineering time

Report **sensitivity** to GPU price, utilization, and volume instead of a single number.

---

## Distillation flavors

| Flavor | What it does | Notes |
|---|---|---|
| **Black-box (sequence-level)** | Train on teacher output text only | Practical default; works with any API model |
| **White-box (logit)** | Match teacher token distributions with KL loss | Richer signal; needs an open teacher |
| **Rationale / reasoning** | Train on reasoning plus answer | Helps harder tasks; drop reasoning at serving time to save tokens |
| **Self-distillation / rejection sampling** | Sample several answers, keep only verified ones | Cleaner data than a single output |

---

## Key challenges and pitfalls

- **The student inherits the teacher's mistakes.** Check against gold labels.
- **Input coverage matters most.** The student only learns what the inputs cover. Include hard and rare cases.
- **Narrowness is the point.** Make the task boundary explicit; consider an off-task guard.
- **More data has diminishing returns.** Use the learning curve before spending more on labels.
- **Label noise.** Self-consistency voting and dropping low-agreement examples often beats adding more data.
- **Fair baselines.** A strong few-shot small model may make distillation unnecessary.
- **Honest cost accounting.** Distillation only wins at sufficient volume.
- **Latency vs. throughput.** Real savings depend on batching and GPU utilization.
- **Drift.** When categories or inputs change, relabel and retrain. A cascade makes drift easier to catch.
- **Cascades often beat pure replacement.** Calibration quality decides how well they work.
- **Evaluation leakage.** Near-duplicates between train and gold inflate results.

---

## Build order

1. Task spec, output schema, and gold evaluation set
2. Teacher accuracy on gold (the ceiling) and cost per request
3. Baselines: prompted small model and classical model (the floor)
4. Teacher labeling on the full input pool, with caching and cost tracking
5. Curation, first student fine-tune, gold evaluation
6. Learning curve over training set size
7. Error analysis, targeted relabeling, retraining
8. Latency, throughput, and cost benchmarking under load
9. Break-even analysis and confidence-based cascade
10. Final report: quality vs. cost vs. latency, with confidence intervals

---

## Variants worth trying

- **Student size sweep:** 0.5B vs. 1.5B vs. 3B vs. 8B, to find the smallest model that clears your quality bar
- **Rationale distillation vs. answer-only:** is the extra token cost worth it?
- **Encoder classifier vs. generative student:** a small BERT-style model can be 10-100x cheaper for classification
- **Multi-teacher ensembles:** keep only agreed answers for cleaner data
- **Structured-output specialist:** train for valid JSON on a specific schema and measure validity rate
- **Active learning loop:** use student uncertainty to choose which inputs get teacher or human labels next

---

## Terms of service and data rights

Before investing time:

- Check whether the teacher provider's terms allow using outputs to train another model.
- Check whether your input data (especially customer data) may be sent to a third-party API at all.
- An open teacher (e.g., Llama or Qwen family) avoids the first problem and enables white-box distillation.

---

## Reference points

- Hinton et al., *Distilling the Knowledge in a Neural Network*: the classic reference
- DistilBERT: the standard worked example
- Hsieh et al., *Distilling Step-by-Step*: rationale-based distillation with small data requirements
- Alpaca, Orca, Zephyr: black-box distillation into open models, worth reading for pitfalls
- Hugging Face **TRL** (`SFTTrainer`) and **peft** for training
- **vLLM** for fast serving and benchmarking
- Hugging Face **distilabel** for generating and filtering synthetic training data

---

## Results

> Fill this in after the final report.

| System | Accuracy (95% CI) | Schema validity | p50 latency | p95 latency | $/1k requests |
|---|---|---|---|---|---|
| Teacher | | | | | |
| Student | | | | | |
| Prompted small | | | | | |
| TF-IDF | | | | | |
| Cascade | | | | | |

---

## License
MIT.