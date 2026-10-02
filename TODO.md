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
- [~] `make labels` for all 7 benchmarks — resuming 2026-10-02
  - [x] gsm8k (3000) · [x] mmlu (3000) · [~] mbpp (973 / 974)
  - [ ] arc_challenge · [ ] math · [ ] humaneval · [ ] bbh
  - [x] **Fixed:** `mbpp-0493` (3,741-token prompt, the only one of 10,430 that cannot fit with 1,024
    output tokens in 4,096) is now excluded before generation via vLLM `/tokenize`, recorded in
    `.meta.json` and the summary; the run exits 0. Next: rerun `make labels BENCH=mbpp`.
- [ ] `make labels-summary`
- [ ] **Gate:** manual review of 30 samples; ≤ 1 mislabel, else fix grader and re-grade
- [ ] Write `reports/R1-labels.md` (truncation rate needs the Phase 2 encoder tokenizer)
- [ ] Verify each dataset licence at source before publishing

## Phase 2 — Router v0 and the kill gate  → `reports/R2-kill-gate.md`
- [ ] Split integrity test (written first, cannot be skipped in CI)
- [ ] Frontier backend + spend guard; frontier answers for val/test prompts, cached
- [ ] Heuristic baseline (token count + keywords → logistic regression)
- [ ] Random baseline at matched escalation rate
- [ ] v0: frozen sentence encoder + logistic regression
- [ ] Shared `predict_proba` interface test
- [ ] Cost model (pinned prices/GPU rate in config) + unit test against hand-computed value
- [ ] Threshold sweep 0→1 step 0.01; monotonicity test; reproducibility test
- [ ] Row-split AUROC (labelled optimistic) alongside dataset-split AUROC
- [ ] Truncation rate for R1
- [ ] **Gate:** v0 dominates heuristic, beats random, AUROC meaningfully > 0.5 — else STOP
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
| 2026-10-02 | Frontier model: Claude Opus 5.5 (`claude-opus-5-5`), $4 / $20 per 1M input / output tokens, pinned in `config.py` | Owner chose Claude; Opus 5.5 is the current default Claude model. Thinking cannot be disabled on it and is billed as output, so cost uses the API's reported usage, never estimates. No sampling controls, so frontier answers are generated once and cached rather than assumed reproducible |
| 2026-10-02 | Prompts that cannot fit `max_model_len` with the fixed 1,024-token budget are excluded and recorded, not failed or labelled | Only `mbpp-0493` qualifies. A smaller budget for one item would break the single decode config; a bigger context window would change the recorded serving setup mid-run |
| 2026-10-02 | Keep HumanEval at all 164 items as a test track | Owner decision. It is below the "several hundred rows" rule, so its per-benchmark figures are reported with confidence intervals, and the code track's test weight also comes from BBH/MATH alongside it. Revisit only if its interval is too wide to say anything |

## Open questions for the owner
- None open.
