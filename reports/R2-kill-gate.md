# R2 — Router v0 and the kill gate

> Status: **in progress.** Router evaluation (AUROC, kill-gate criterion 3) is generated below by
> `make train-v0`. Criteria 1 and 2 need the frontier model's answers and the cost-quality sweep.
> Do not edit between the GENERATED markers.

## 1. What was measured

Whether a router reading only the prompt can tell which prompts the local model answers
correctly, on benchmarks it never saw in training, and whether it does better than a
length-and-keyword heuristic and than random routing.

## 2. Setup

- Labels: `data/labels/Qwen__Qwen2.5-1.5B-Instruct.parquet` (R1).
- Split by dataset: train GSM8K, MMLU, MBPP · validation ARC-Challenge · test MATH, HumanEval, BBH.
  `make check` includes a test that fails if any prompt appears in more than one split.
- Routers, all behind one `predict_proba` interface:
  - **heuristic** — logistic regression on two features: log(1 + word count) and the number of
    keywords from a fixed list (`router/baselines.py`) present in the prompt.
  - **random** — a uniform score from a hash of the prompt, ignoring its content.
  - **v0** — frozen `BAAI/bge-base-en-v1.5` CLS embeddings + logistic regression; C chosen on
    validation AUROC from {0.01, 0.1, 1, 10}.
- Kill-gate criterion 3 was fixed before the first run: v0's 95% bootstrap interval for test
  AUROC must have its lower end above 0.55.

## 3. Results

<!-- BEGIN GENERATED: router-v0 -->
`TBD` — run `make train-v0`.
<!-- END GENERATED: router-v0 -->

Cost-quality curve, matched-escalation comparison with random, gate criteria 1–2: `TBD`
(Phase 2, after the frontier answers).

## 4. What surprised me

`TBD` — written after the full gate.

## 5. What this changes

`TBD` — written after the full gate.
