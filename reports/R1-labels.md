# R1 — Labels

> Status: **complete — manual review gate passed (round 2).** The tables between the GENERATED
> markers are written by `make labels-summary` from `data/labels/`; do not edit them by hand.
> Every figure quoted in the prose below appears in `results/labels-summary.json` or in the run
> history in `data/labels/Qwen__Qwen2.5-1.5B-Instruct.meta.json`.

## 1. What was measured

How often the local model answers each benchmark correctly under fixed greedy decoding, graded by
a deterministic per-benchmark grader. These labels are what the router learns to predict.

## 2. Setup

- **Local model:** `Qwen/Qwen2.5-1.5B-Instruct` at commit `989aa79`, served by vLLM 0.30.0
  (bf16, `max_model_len` 4096, `max_num_seqs` 32, `gpu_memory_utilization` 0.85).
- **Decoding:** greedy (temperature 0), `top_p` 1, seed 0, at most 1,024 new tokens, identical
  for every benchmark.
- **Hardware:** laptop NVIDIA RTX 4060 Laptop GPU (8 GB), WSL2 Ubuntu.
- **Split** (`data/splits.yaml`): train GSM8K, MMLU, MBPP · val ARC-Challenge · test MATH,
  HumanEval, BBH.
- **Code grading:** every HumanEval/MBPP program ran in the Docker sandbox (no network, capped
  resources, non-root, read-only, 10 s kill), image pinned by digest.
- **Grader self-check** (`make grader-selfcheck`): an answer built from each item's own reference
  (or the benchmark's canonical solution for code) graded correct for **all 10,430 items, 0
  failures**.

## 3. Results

<!-- BEGIN GENERATED: labels-summary -->
Model: `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa7980e` · decoding: `{"max_tokens": 1024, "seed": 0, "temperature": 0.0, "top_p": 1.0}`

**Overall base rate: 60.0%** (6259 / 10429 correct). Every later accuracy is read against this.

| Benchmark | Role | n | Correct | Accuracy | Excluded | Hit token cap | Mean out tokens | Flag |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| arc_challenge | val | 1172 | 806 | 68.8% | 0 | 0.0% | 197 |  |
| bbh | test | 1620 | 622 | 38.4% | 0 | 0.5% | 233 |  |
| gsm8k | train | 3000 | 2428 | 80.9% | 0 | 0.1% | 257 |  |
| humaneval | test | 164 | 81 | 49.4% | 0 | 0.0% | 196 |  |
| math | test | 500 | 239 | 47.8% | 0 | 9.2% | 557 |  |
| mbpp | train | 973 | 449 | 46.1% | 1 | 0.0% | 231 |  |
| mmlu | train | 3000 | 1634 | 54.5% | 0 | 0.2% | 250 |  |

| Role | n | Base rate |
| --- | ---: | ---: |
| train | 6973 | 64.7% |
| val | 1172 | 68.8% |
| test | 2284 | 41.2% |

Tokens generated: 2,663,977 · generation wall-clock: 0.48 h

**Excluded items (1)** — never generated or labelled:
- mbpp / `mbpp-0493`: prompt_exceeds_context: 3741 prompt + 1024 output > 4096 max_model_len
<!-- END GENERATED: labels-summary -->

Truncation rate at the router encoder's limit: `TBD` (needs the Phase 2 encoder tokenizer).

### Manual review gate

| Round | Sample | Mislabels | Verdict | Sheet |
| --- | --- | ---: | --- | --- |
| 1 | 30 random items, seed 0 | 2 | **Failed** (allows ≤ 1) | `reports/review/R1-sample-round1.md` |
| 2 | 30 fresh random items, seed 1, after the fix | 1 | **Passed** | `reports/review/R1-sample.md` |

- **Reviewer:** AI assistant at the owner's request, 2026-10-02. An owner spot-check of both
  sheets is recommended.
- **Round 1 misses:** two correct answers graded wrong because they were not in the requested
  form — a multiple-choice reply that concluded with a bolded option line instead of "The answer
  is (X)", and a BBH yes/no question answered with "Jim does not tell the truth".
- **Scan of all labels:** both patterns were systematic (no letter extracted from 108 of 4,172
  multiple-choice replies; 113 of 208 wrong BBH yes/no-type answers did not start with the answer
  word). The graders were fixed with fixtures for each form, including negative cases.
- **Round 2 miss:** a correctly ordered word list written with commas and capitals. Fixed; the
  complete set of labels it affected (4 word_sorting items) was checked.
- **Effect of the fixes**, from the regrade entries in the metadata: MMLU 1,613 → 1,634,
  ARC-Challenge 778 → 806, BBH 573 → 622; GSM8K, MATH, HumanEval and MBPP unchanged. Every change
  was wrong → correct; **no label went from correct to wrong**.

## 4. What surprised me

- **Answer format, not knowledge, was the main grading risk.** The model often answered correctly
  but ignored the requested final-line template. A strict template grader would have understated
  the local model and pushed the router toward escalating. The review gate caught this before any
  router was trained on the labels — which is exactly what the gate is for.
- **The test benchmarks are much harder for the local model than the training ones:** base rate
  41.2% on test versus 64.7% on train and 68.8% on validation. The router will be evaluated on a
  distribution with far more prompts it should escalate than it saw in training.
- **MATH is the only track where the 1,024-token budget binds:** 9.2% of MATH generations hit the
  cap (0.5% or less everywhere else), with the longest mean output (557 tokens).
- **Degenerate generations exist** — e.g. a word-sorting reply that repeated one word a hundred
  times. They are graded wrong, which is correct, and they are prompts the router should learn to
  escalate.
- **Labelling was quick on a laptop GPU:** 2,663,977 tokens in 0.48 h of generation wall-clock.
- **Only one prompt (`mbpp-0493`) could not fit** the 4,096-token context with the answer budget;
  it is excluded and recorded rather than labelled.

## 5. What this changes

- **The gate passed; Phase 2 can start.** The overall base rate, **60.0%**, is the trivial
  baseline for the classifier: a router that always says "local" is 60.0% accurate and useless.
  Every later accuracy is read against it.
- **No benchmark is near-constant** (none above 95% or below 5%), so all seven carry routing signal.
- **Calibration must be checked on test, not only on validation.** Temperature scaling will be
  fitted on ARC-Challenge (68.8% base rate) but applied to test benchmarks at 41.2%; Phase 3 will
  report ECE on both so a calibration that does not transfer is visible.
- **The 1,024-token budget stays.** Changing it would invalidate every label; MATH's cap hits are
  part of what "the local model cannot answer this within budget" means, and the limitation is
  stated in the README.
- **Known residual strictness:** a yes/no BBH answer given as a sentence with no answer word
  (e.g. "Jen did not intend to kill the puppies") is still graded wrong. That errs toward
  escalation, the safe direction, and is documented in the grader tests.
- **Still open:** truncation rate at the encoder limit (Phase 2) and licence verification at
  source before publishing.
