# R2 — Router v0 and the kill gate

> Status: **kill gate FAILED on criterion 3 — building is stopped pending the owner's decision.**
> v0's held-out AUROC interval does not clear the pre-registered floor, and v0 is significantly
> *worse* than the length-and-keyword heuristic. Per 11-roadmap, no serving infrastructure is
> built around this result. Tables between the GENERATED markers are written by `make train-v0`
> from `results/router-v0.json`; do not edit them by hand.

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
Encoder `BAAI/bge-base-en-v1.5` @ `a5beb1e3e6` (frozen for v0) · seed 0 · split sizes train 6973, val 1172, test 2284

**Dataset-level split** (the headline): routers trained on train benchmarks, C chosen on validation, scored on held-out test benchmarks. 95% bootstrap intervals.

| Router | Val AUROC | Test AUROC | Test 95% CI | Test ECE | Test Brier |
| --- | ---: | ---: | --- | ---: | ---: |
| heuristic | 0.523 | **0.606** | [0.581, 0.629] | 0.243 | 0.293 |
| random | 0.503 | **0.507** | [0.485, 0.532] | 0.248 | 0.329 |
| v0 | 0.559 | **0.549** | [0.524, 0.572] | 0.186 | 0.278 |

Test base rate: 41.2% (the local model is right this often).

**v0 against the baselines on test** (paired bootstrap):

| Comparison | AUROC difference | 95% CI |
| --- | ---: | --- |
| v0 minus heuristic | -0.058 | [-0.090, -0.025] |
| v0 minus random | +0.042 | [0.008, 0.074] |

**Per test benchmark** (where does prompt-only routing work?):

| Benchmark | n | Base rate | Heuristic | Random | v0 | v0 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bbh | 1620 | 38.4% | 0.575 | 0.517 | 0.520 | [0.489, 0.550] |
| humaneval | 164 | 49.4% | 0.707 | 0.427 | 0.606 | [0.518, 0.688] |
| math | 500 | 47.8% | 0.685 | 0.497 | 0.643 | [0.594, 0.692] |

**Row-level split — OPTIMISTIC, for comparison only** (random 80/20 over all rows; the router can learn which benchmark a prompt came from):

| Router | Row-split AUROC | Dataset-split AUROC | Gap |
| --- | ---: | ---: | ---: |
| heuristic | 0.581 | 0.606 | -0.025 |
| random | 0.506 | 0.507 | -0.000 |
| v0 | 0.712 | 0.549 | +0.164 |

**Diagnostics** — fit on the training benchmarks, and mean score per benchmark against the local model's actual base rate:

| Router | Train AUROC (in-sample) | Test AUROC, pooled | Test AUROC, within-benchmark |
| --- | ---: | ---: | ---: |
| heuristic | 0.606 | 0.606 | 0.608 |
| random | 0.491 | 0.507 | 0.507 |
| v0 | 0.755 | 0.549 | 0.553 |

| Benchmark | Role | Base rate | Heuristic mean score | v0 mean score |
| --- | --- | ---: | ---: | ---: |
| gsm8k | train | 0.809 | 0.666 | 0.793 |
| mbpp | train | 0.461 | 0.682 | 0.478 |
| mmlu | train | 0.545 | 0.617 | 0.556 |
| arc_challenge | val | 0.688 | 0.656 | 0.653 |
| bbh | test | 0.384 | 0.634 | 0.608 |
| humaneval | test | 0.494 | 0.619 | 0.520 |
| math | test | 0.478 | 0.725 | 0.595 |

Prompts longer than the encoder's 512-token limit: 22 (0.21%).

**Kill gate, criterion 3** (v0 test AUROC 95% CI low > 0.55): low = 0.524 → **FAIL**. Criteria 1 and 2: pending the frontier answers and the cost-quality sweep.
<!-- END GENERATED: router-v0 -->

Cost-quality curve, matched-escalation comparison with random, gate criteria 1–2: `TBD` — they
need the frontier model's answers on the test prompts. Criterion 3 already fails, so the gate
fails whatever they show; the curve is still needed for the write-up of either outcome.

### Gate verdict

| Criterion (fixed before the run) | Result |
| --- | --- |
| 1. v0's cost-quality curve dominates the heuristic's | `TBD` (needs frontier answers) — v0's AUROC is below the heuristic's, so this is unlikely |
| 2. v0 beats random routing at matched escalation | `TBD` (needs frontier answers) |
| 3. v0 test AUROC: lower end of 95% interval > 0.55 | **FAIL** (see the generated block) |

**Verdict: FAIL.** Following 11-roadmap: stop building; the options are to change the benchmark
mix, change the local model, or write up the negative result.

## 4. What surprised me

- **The frozen encoder lost to the two-feature heuristic** on held-out benchmarks, and the paired
  difference interval is entirely below zero, so this is not noise.
- **v0 learned where a prompt came from, not how hard it is.** On the training benchmarks its
  mean score lands almost on each benchmark's base rate and its in-sample AUROC is far higher than
  on test (diagnostics table). With only three training benchmarks, "which benchmark is this?" is
  the easiest pattern for an embedding classifier to pick up.
- **The row split would have hidden this completely.** On a random row split v0 looks clearly
  better than on the dataset split, while the heuristic barely changes. This is exactly the
  inflation 04-datasets warned about, and the reason the dataset-level split is non-negotiable.
- **Within each test benchmark v0 also ranks prompts worse than the heuristic**, so the problem is
  not only the ordering *between* benchmarks: frozen general-purpose embeddings capture topic far
  more than difficulty.
- **The heuristic is weak too.** Its held-out AUROC is modest and its validation AUROC is barely
  above chance; it beats v0, but neither is a strong router yet.
- **Only 22 prompts (0.21%) exceed the encoder's 512-token limit**, so truncation is not a factor.

## 5. What this changes

- **No serving work (Phase 4) is started.** The roadmap is explicit that infrastructure is not
  built around a result that does not exist.
- **The frontier answers are still worth collecting.** They depend only on the test prompts, not
  on the router or the local model, so they are needed for the cost-quality curve whichever option
  is chosen, including a negative-result write-up.
- **Options for the owner** (any second attempt is reported *alongside* this one, never instead of
  it, and keeps the same held-out test benchmarks):
  1. **Write up the negative result** — "prompt-only routing with a frozen encoder learns benchmark
     provenance, not difficulty" — with the cost curve for the heuristic and always-frontier.
  2. **Change the benchmark mix** — train on many more, more varied benchmarks so that benchmark
     identity stops being a usable shortcut, and/or re-weight training so every benchmark has the
     same share of correct and incorrect examples. Labelling is cheap (about half an hour on the
     laptop GPU for the current set).
  3. **Change the local model** — does not address the provenance shortcut by itself.
  4. **Go straight to the fine-tuned v1** — not recommended without option 2: fine-tuning on the
     same three benchmarks would most likely learn provenance even more strongly.
