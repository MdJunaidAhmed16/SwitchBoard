# Switchboard — build tracker

Source of truth for scope: the planning docs (`project-docs/`, kept local).
Each phase ends with a report in `reports/` and a gate that can fail.
`[x]` done · `[ ]` open · `[~]` in progress / needs the owner to run something.

---

## Stage 0 — Repository foundation

- [x] `git init`, `.gitignore` (incl. `project-docs/`, secrets, raw data, cache, tf state), `.gitattributes` (LF)
- [x] `TODO.md` tracker
- [x] `pyproject.toml` — Python 3.11, exact-pinned deps, ruff (line 100), mypy strict, pytest markers
- [x] `uv.lock` committed
- [x] `src/switchboard/config.py` — single env-driven `Settings` object
- [x] `src/switchboard/log.py` — structlog JSON, configured explicitly (no import side effects)
- [x] `Makefile` — install, lint, format, typecheck, test, check
- [x] Pre-commit: ruff, gitleaks secret scan, whitespace/yaml checks
- [x] CI (GitHub Actions): lint, type check, unit tests, secret scan
- [x] `.env.example` (no real values)
- [x] README skeleton with the thesis, both baselines and **falsification conditions written before any training code**; every number `TBD`

## Phase 1 — Labels  → `reports/R1-labels.md`

### Code
- [x] Benchmark registry: pinned HF revisions, fixed seeded sampling, prompt templates
- [x] Router input (`router_text`) kept separate from the generation prompt (no format-instruction leakage)
- [x] `data/splits.yaml` — train / val / test at dataset level
- [x] Graders (pure functions, `(prediction, reference) -> bool`)
  - [x] GSM8K — numeric exact match with normalisation
  - [x] MATH — `\boxed{}` extraction + LaTeX normalisation + numeric equivalence
  - [x] MMLU / ARC — multiple-choice letter extraction
  - [x] BBH — exact match, MC-letter and numeric aware
  - [x] HumanEval / MBPP — unit-test execution in a sandbox
- [x] Sandbox: Docker, `--network none`, memory/CPU/pids caps, non-root, read-only FS, timeout; image pinned by digest
- [x] Generation cache (SQLite) keyed `(model_id, prompt_hash, decode_params_hash)` — resumable, never regenerates
- [x] vLLM client (OpenAI-compatible chat endpoint), bounded concurrency, retry with backoff
- [x] `make labels BENCH=<name>` → `data/labels/<model>.parquet` (+ `.meta.json` with versions, decode params, wall-clock)
- [x] Label summary → `results/labels-summary.json` + markdown table (accuracy/base rate/flags from data, never typed)
- [x] Review sampler → `reports/review/R1-sample.md` (30 random triples)
- [x] Grader self-check over every real item (`make grader-selfcheck`) — 0 failures on 9,292 non-code items; 1,138 canonical code solutions pass assembly

### Tests (13-testing-and-reports, Phase 1)
- [x] Grader fixtures ≥ 10 per grader incl. adversarial (extra prose, wrong format, empty, refusal)
- [x] Grader normalisation: `"42"`, `"42.0"`, `" 42 "`, `"The answer is 42"` agree
- [x] Sandbox isolation (command has no network + timeout; live checks when Docker present)
- [x] Cache key stability
- [x] Label schema (required columns, no nulls)
- [x] Pipeline end-to-end against a fake vLLM (resume never regenerates, merge per benchmark)
- [x] Determinism (integration, live vLLM) — `make test-integration`, 3 passed
- [x] Code-grader live execution in Docker — `make test-docker`, 6 passed

### Run (owner, on the GPU)
- [x] Move the repo into the WSL filesystem (`~/projects/switchboard`), `make install`
- [x] Install Docker Engine in WSL; `make sandbox-pull`; `make test-docker` (6 passed)
- [x] `make grader-selfcheck` with code tracks in Docker — 0 failures on all 7 benchmarks (10,430 items, ~6 min)
- [x] Confirm the local model (Qwen2.5-1.5B-Instruct)
- [x] Start vLLM (`make vllm`, v0.30.0); `make test-integration` (3 passed)
- [x] Smoke run: `make labels BENCH=gsm8k LIMIT=20` and `BENCH=mbpp LIMIT=10` — 0 failed; all wrong labels inspected and genuinely wrong
- [x] `make labels` for all 7 benchmarks — 10,429 labelled, 1 excluded (`mbpp-0493`)
  - [x] gsm8k · [x] mmlu · [x] mbpp · [x] arc_challenge · [x] math · [x] humaneval · [x] bbh
  - [x] **Fixed:** `mbpp-0493` (3,741-token prompt, the only one of 10,430 that cannot fit with 1,024
    output tokens in 4,096) is now excluded before generation via vLLM `/tokenize`, recorded in
    `.meta.json` and the summary; the run exits 0.
- [x] `make labels-summary` — overall base rate 60.0% (see `reports/R1-labels.md`)
- [x] **Gate:** manual review of 30 samples; ≤ 1 mislabel, else fix grader and re-grade
  - [x] Round 1 (seed 0): 2 mislabels → **failed**; both systematic format misses
  - [x] Graders fixed (MC: boxed letter, "correct choice is", concluding option line; BBH: answer word anywhere in final clause, truth-teller statements, bold "Final Answer"); `make regrade` added; 94 labels wrong→correct, 0 correct→wrong
  - [x] Round 2 (fresh, seed 1): 1 mislabel → **passed**; the miss (word list with commas) fixed too, all 4 affected labels checked
- [x] Write `reports/R1-labels.md` (truncation rate deferred to Phase 2, needs the encoder tokenizer)
- [ ] Owner spot-check of the two review sheets (review was done by an AI assistant at the owner's request)
- [x] Merge `feat/labeling-pipeline` into `main` — fast-forward, all commits kept; CI green on `main`
- [ ] Verify each dataset licence at source before publishing

## Phase 2 — Router v0 and the kill gate  → `reports/R2-kill-gate.md`
- [x] Split integrity test (written first, runs on the committed labels, not skippable) — no prompt crosses splits
- [x] Heuristic baseline (log word count + keyword count → logistic regression; keyword list fixed before evaluation)
- [x] Random baseline (hash-based uniform score; matched-escalation comparison happens in the sweep)
- [x] v0: frozen `BAAI/bge-base-en-v1.5` + logistic regression, C chosen on validation
- [x] Shared `predict_proba` interface test (shape, range, order independence)
- [x] Row-split AUROC (labelled optimistic) alongside dataset-split AUROC
- [x] Truncation rate for R1 — 22 prompts (0.21%) over 512 tokens
- [x] `make train-v0` → `results/router-v0.json`, scores parquet, generated R2 tables; reproducible across reruns
- [x] **Gate criterion 3 (pre-registered: v0 test AUROC 95% CI low > 0.55): FAILED.** v0 also significantly below the heuristic. Diagnostics show v0 learned benchmark provenance
- [x] **Owner decision after the failed gate (2026-10-05):** attempt 2 — change the benchmark mix + re-weight training — and also evaluate the fine-tuned v1. Serving (Phase 4) still not started
- [x] Attempt 2 pre-registered in R2 before any attempt-2 result (`0154ccf`)
- [x] Six training benchmarks added (GSM-Hard, SVAMP, AQuA-RAT, CommonsenseQA, MedMCQA, QASC; 8,016 prompts); grader self-check 0 failures
- [x] Attempts defined in code (`router/attempts.py`); benchmark-balanced weights; attempt 1 still reproduces exactly (`make train-v0`)
- [x] v1 fine-tuned router + temperature scaling (`router/v1.py`, `router/calibrate.py`); overfit-one-batch, validation-only calibration and seed-determinism tests
- [x] `make train-attempt2`: heuristic, random, v0 and v1 ×3 seeds with the pre-registered comparisons
- [x] Label the six new benchmarks — 8,016 labels, 0 failed; unparsable-answer scan: residual answer-by-text misses ≤ 11 (0.06%)
- [ ] Owner spot-check of the new benchmarks' labels (no separate 30-item round was run)
- [x] `make train-attempt2` (54 min; length-grouped batching + per-seed checkpoints after the first run was lost) — written up in R2 beside attempt 1
- [x] **Attempt 2 verdict:** provenance shortcut removed; v0 still fails criterion 3; **v1 passes criterion 3 on all seeds and beats v0 beyond the seed spread, but ties the heuristic** (paired intervals include 0); calibration improves but test ECE stays above 0.05
- [x] Frontier backend via OpenRouter (OpenAI-compatible, same client shape as vLLM) + spend guard that refuses before sending; `make frontier-pilot`, `make frontier MAX_USD=…`
- [x] Frontier pilot: 30 prompts, $0.23 (computed cost matches OpenRouter's charge exactly), projected full test set ≈ $14.75
- [ ] **Owner approval** for the full frontier run (`make frontier MAX_USD=20`) — needed for gate criteria 1–2 and the cost curve
- [ ] Cost model (pinned prices/GPU rate in config) + unit test against hand-computed value
- [ ] Threshold sweep 0→1 step 0.01; monotonicity test (at-threshold monotonicity ✓ in unit tests); reproducibility test
- [ ] Gate criteria 1–2 (curve vs heuristic, vs random at matched escalation)
- [ ] README results section filled from the results file, even if bad

## Phase 3 — Router v1  → `reports/R3-router-v1.md`
- [ ] Fine-tuned encoder (BCE, AdamW, warmup, bf16, early stop on val AUROC), 3 seeds
- [ ] Overfit-one-batch test; seed-variance test; no-leakage test
- [ ] Temperature scaling; ECE (10 bins), Brier, reliability diagram before/after
- [ ] τ chosen by a quality floor fixed in advance
- [ ] **Gate:** v1 beats v0 beyond seed spread and ECE < 0.05 — else ship v0 and say so

## Phase 4 — Serving  → `reports/R4-serving.md`
- [ ] ONNX export + Triton model repo + parity test (≤ 1e-4) in CI
- [ ] Gateway: `/route`, `routing.py` as the only τ comparison, decision records
- [ ] Failure handling: router/vLLM down → escalate; frontier errors → retry then surface
- [ ] docker compose: Triton (CPU) + vLLM + gateway
- [ ] Load generator: concurrency 1/8/32, 10% warmup discarded
- [ ] **Gate:** router p99 ≤ 15 ms, gateway ≤ 5 ms, batching on/off chart, all requests logged

## Phase 5 — Infrastructure  → `reports/R5-reproducibility.md`
- [ ] Terraform: serving-cpu, serving-gpu, storage, budget modules; GCS state backend
- [ ] `make gpu-up` / `gpu-down`; `make bench` with teardown trap
- [ ] Idle watchdog (30 min); budget alerts 50/90/100%
- [ ] **Gate:** clean-clone reproduction; crashed run leaves no GPU alive

## Phase 6 — Demo and write-up  → `reports/R6-final.md`
- [ ] Live scoring page with threshold slider; cached replay clearly labelled
- [ ] Static dashboard reading `results/latest.json`
- [ ] Rate limit, prompt cap, no anonymous escalation, daily ceiling
- [ ] README number-provenance check in CI
- [ ] **Gate:** demo works with GPU off; no unprovenanced number

---

## Decisions log

Decisions made while building, recorded here so they do not live only in chat.

| Date | Decision | Why |
| --- | --- | --- |
| 2026-09-30 | Local model `Qwen/Qwen2.5-1.5B-Instruct` @ `989aa79` (confirmed by owner 2026-10-02) | 8 GB 4060: a 3B bf16 model (~6.2 GB weights) leaves almost no KV cache; 1.5B is Apache-2.0 while Qwen2.5-3B uses a non-commercial research licence |
| 2026-09-30 | Validation benchmark: ARC-Challenge (test split) | Docs require validation on a held-out *benchmark*, but no val benchmark was named; ARC is public, MC-gradable, and not in train or test |
| 2026-09-30 | MATH via `HuggingFaceH4/MATH-500` | The original `hendrycks/competition_math` repo was taken down; MATH-500 is the standard 500-problem subset with extracted answers |
| 2026-09-30 | Router input is the task text only; answer-format instructions are appended only for generation | Format suffixes differ per track and would let the router learn provenance instead of difficulty |
| 2026-09-30 | One decode config for every benchmark: greedy (T=0), seed 0, max 1024 new tokens | One `decode_params_hash`; a generation cut off at the cap is graded as the model failing |
| 2026-09-30 | Datasets fetched as pinned parquet/jsonl via `huggingface_hub`, not `datasets.load_dataset` | Exact revision pinning, fewer dependencies, no loader scripts |
| 2026-09-30 | vLLM treated as an external server; not a project dependency | vLLM pins its own torch/CUDA; keeping it out of the project env avoids conflicts with training deps |
| 2026-09-30 | Code sandbox: one throwaway container per program, program piped via stdin (no volume mount) | Nothing on the host filesystem is exposed to model-generated code |
| 2026-10-02 | Frontier model: Claude Opus 5.5, $4 / $20 per 1M input / output tokens, pinned in `config.py` (provider changed to OpenRouter on 2026-10-03, see below) | Owner chose Claude; Opus 5.5 is the current default Claude model. Thinking cannot be disabled on it and is billed as output, so cost uses the API's reported usage, never estimates. No sampling controls, so frontier answers are generated once and cached rather than assumed reproducible |
| 2026-10-02 | Prompts that cannot fit `max_model_len` with the fixed 1,024-token budget are excluded and recorded, not failed or labelled | Only `mbpp-0493` qualifies. A smaller budget for one item would break the single decode config; a bigger context window would change the recorded serving setup mid-run |
| 2026-10-02 | Grade answers given in non-template forms (boxed letter, concluding option line, answer word anywhere in the final clause, truth-teller statements, word lists ignoring commas/case) | The R1 review found the strict template undercounted correct answers systematically; every resulting label change was wrong→correct and audited |
| 2026-10-03 | Frontier calls go through **OpenRouter** (`anthropic/claude-opus-5.5`, $4 / $20 per 1M listed), key from `OPENROUTER_API_KEY` in the gitignored `.env` | Owner's choice of key provider. OpenRouter's API is OpenAI-compatible, so the frontier and vLLM clients share one shape (as 06-serving asks); a per-key credit limit set in OpenRouter adds a hard spend ceiling outside the code |
| 2026-10-03 | Every step lands through a **pull request** (template in `.github/`), merged with a merge commit — never squash; superseded the direct fast-forward used for Phase 1 | Owner wants PRs in the history; CI runs on each PR; a merge commit keeps every commit and its real date, so the contribution graph still reflects the work |
| 2026-10-03 | Merge feature branches into `main` by **fast-forward**, not squash | Owner wants the commit history (and GitHub's contribution graph) to show the real work; squashing collapses a branch into one commit. Commits are never backdated |
| 2026-10-03 | Router encoder `BAAI/bge-base-en-v1.5` @ `a5beb1e` for both v0 (frozen) and v1 (fine-tuned) | 110M, MIT, 512 tokens, strong frozen embeddings and fine-tunable; one backbone makes v1's gain over v0 the value of fine-tuning alone |
| 2026-10-03 | Kill-gate criterion 3 made concrete before the first run: lower end of v0's 95% bootstrap interval for test AUROC > 0.55 | "Meaningfully above 0.5" needs a number fixed in advance, or it can be argued after the fact |
| 2026-10-03 | huggingface-hub 2.0.0 → 1.33.0 | tokenizers 0.23 (needed by transformers 5.18) requires hub < 2.0; the download calls used are unchanged |
| 2026-10-02 | Keep HumanEval at all 164 items as a test track | Owner decision. It is below the "several hundred rows" rule, so its per-benchmark figures are reported with confidence intervals, and the code track's test weight also comes from BBH/MATH alongside it. Revisit only if its interval is too wide to say anything |

## Open questions for the owner
- None open.
