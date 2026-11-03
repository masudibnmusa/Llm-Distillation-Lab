# Task spec

**Task:** Classify a customer-support message into exactly one category.

**Input:** free text, 1-500 tokens, English.
**Output:** JSON `{"category": "<label>"}` where label is one of `data/labels.json`.

**Success criteria**
- Student accuracy within 2 points of teacher on the gold set (95% CI reported)
- Schema validity >= 99%
- p95 latency < 200 ms, cost per 1k requests < 10% of teacher

**Out of scope:** anything that is not a support message. The student must not be used as a general assistant;
off-task inputs should be routed to the teacher or rejected (see robustness / cascade).

**Gold set:** 200-500 human-verified examples, never used for training or few-shot prompts.