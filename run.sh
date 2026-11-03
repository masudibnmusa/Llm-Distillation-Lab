#!/usr/bin/env bash
set -euo pipefail
D="python -m distill.cli"

# 1. Inputs (Banking77 as example; check column names for your HF dataset)
$D collect --hf mteb/banking77 --split train --text-col text --label-col label_text --limit 10000

# 2. Gold set (use dataset labels as gold; or run `$D gold --n 300` to label by hand)
$D gold --n 300 --use-source-labels
$D split

# 3. Teacher ceiling + baselines
$D eval --system teacher
$D eval --system prompted_small

# 4. Label the pool (3 samples + majority vote)
$D label --samples 3
$D curate --min-agreement 0.67

# 5. Floor + student
$D eval --system tfidf
$D train --student qwen-1.5b --config distill/student/configs/lora.yaml --tag qwen-1.5b
$D eval --system student --model qwen-1.5b --run-name qwen-1.5b

# 6. Learning curve
for n in 500 1000 2000 5000; do
  $D curate --min-agreement 0.67 --n-train $n
  $D train --student qwen-1.5b --config distill/student/configs/lora.yaml --tag lc_$n --n-train $n
  $D eval --system student --model lc_$n --run-name lc_$n
done

# 7. Analysis, benchmarks, report
$D robust --system student --model qwen-1.5b
$D cascade --student-run qwen-1.5b
$D bench latency --system student --model qwen-1.5b --run-name qwen-1.5b
$D breakeven --gpu-hourly 1.0 --student-rps 20
$D report