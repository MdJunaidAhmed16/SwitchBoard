# What's done — Switchboard build log

A plain-language account of everything built so far: what the project is, how it is put together,
every command we ran and why, and how each design choice follows the project's rules. It is kept
up to date as the project moves through its phases; the newest work is in the
[progress log](#11-progress-log) at the end.

> **Numbers in this file.** Counts such as "10,430 prompts" or "143 tests" describe the build itself
> and can be checked by re-running the command named next to them. Measured *results* (accuracies,
> base rates, cost savings) are never quoted here from memory: they live in
> `results/labels-summary.json` and the generated tables in `reports/`, and are copied into this
> file only after those files exist.

---

## Contents

1. [The idea in one page](#1-the-idea-in-one-page)
2. [Where we are](#2-where-we-are)
3. [Architecture](#3-architecture)
4. [Phase 1 — how labels are made](#4-phase-1--how-labels-are-made)
5. [Every make command, explained](#5-every-make-command-explained)
6. [Other commands we ran](#6-other-commands-we-ran)
7. [How the project's principles are followed](#7-how-the-projects-principles-are-followed)
8. [Problems we hit and how they were fixed](#8-problems-we-hit-and-how-they-were-fixed)
9. [Decisions and why](#9-decisions-and-why)
10. [Git history](#10-git-history)
11. [Progress log](#11-progress-log)

---

## 1. The idea in one page

**Problem.** Most LLM traffic is not hard. Short factual questions, easy arithmetic and boilerplate
code are still sent to an expensive frontier model, because deciding per request is work nobody
wants to do. That default cost grows linearly with traffic.

**Claim.** A small classifier can read a prompt — before anything is generated — and predict
whether a cheap, self-hosted model will answer it correctly. Send the prompt to the cheap model
when that prediction is confident, and to the frontier model otherwise. Most of the frontier
model's quality should be kept at a fraction of the cost.

**The deliverable is a curve, not a number.** For every quality level you might want, what does
the traffic cost? The router outputs a probability `p`; a threshold `τ` turns it into a decision.
Sliding `τ` traces the curve.

**What it is measured against** (always in the same run, on the same axes):

| Reference | What it is | Why it matters |
| --- | --- | --- |
| Always-frontier | Every prompt goes to the frontier model | The quality and cost ceiling; quality is reported as a % of this |
| Length + keyword heuristic | Prompt length plus words like "prove" / "implement", fitted with logistic regression | What a sceptic says you could build in an afternoon without ML |
| Random routing | Escalate the same fraction of prompts at random | If this matches the router, the router learned nothing |

**How the project can fail — written down before any training code** (also in `README.md`):

1. If the heuristic's curve matches or beats the learned router's, the model is unnecessary.
2. If random routing at the same escalation rate matches the router, there is no real signal.

Either outcome gets reported honestly as the headline. A clearly reported negative result is a
stronger artefact than a vague win.

**The models.**

| Role | Model | Where it runs | Cost basis |
| --- | --- | --- | --- |
| Local ("cheap") model | `Qwen/Qwen2.5-1.5B-Instruct`, pinned to commit `989aa79` | Your RTX 4060 laptop GPU, served by vLLM | GPU time |
| Frontier model | Claude Opus 5.5 (`anthropic/claude-opus-5.5`) | Anthropic's model, called through OpenRouter | $4 / $20 per 1M input / output tokens, pinned in `config.py` |
| Router | Small text encoder + classifier (built in Phases 2–3) | CPU, served by NVIDIA Triton (Phase 4) | ~free |

---

## 2. Where we are

| Phase | What it produces | Status |
| --- | --- | --- |
| 0 — Foundation | Repo, tooling, config, CI, README skeleton | ✅ Done |
| 1 — Labels | For every benchmark prompt: did the local model get it right? | ✅ Done — 10,429 labels, review gate passed (round 2); merge into `main` pending |
| 2 — Router v0 + kill gate | Frozen encoder + baselines + first cost/quality curve | Not started |
| 3 — Router v1 | Fine-tuned, calibrated router, 3 seeds | Not started |
| 4 — Serving | ONNX → Triton, gateway, load tests | Not started |
| 5 — Infrastructure | Terraform on GCP, budget guards, one-command reproduction | Not started |
| 6 — Demo + write-up | Live page, dashboard, final README | Not started |

The detailed checklist is `TODO.md`. Phase 2 is a **kill gate**: if the first router does not beat
the heuristic and random baselines, we stop and write up the negative result instead of building
serving infrastructure around a result that doesn't exist.

---

## 3. Architecture

### 3.1 The finished system (target, Phases 4–6)

```
client
  │
  ▼
gateway (FastAPI, CPU)
  │  1. receives prompt
  │  2. asks the router for p = P(local model answers correctly)
  │  3. compares p with threshold τ            ← the ONLY place τ is applied
  │  4. dispatches and writes a decision record
  │
  ├── p ≥ τ ──▶ vLLM (GPU)          small self-hosted model
  └── p <  τ ──▶ frontier API       Claude Opus 5.5, paid per token
          │
          ▼
     router (Triton, CPU) — one forward pass, no generation
```

Each part knows only what it must: the router does not know what happens with its score, the
backends do not know they are part of a routing system, and the gateway is the only component
that sees both paths.

**Why two serving systems?** The router is one short forward pass per request — fixed cost,
latency-critical, batches perfectly — which is exactly what Triton's dynamic batcher is for. The
local model generates token by token with unpredictable lengths; its limit is KV-cache memory,
which is what vLLM's paged attention and continuous batching solve. One system for both would do
one of the two jobs badly.

**Failure defaults all lean toward escalation.** Router unreachable → frontier. vLLM unreachable →
frontier. Escalating costs a few cents; a wrong local answer ships a wrong answer to a user.

### 3.2 What exists today

Phase 1 is an offline pipeline. Nothing is served to users yet.

```
src/switchboard/
├── config.py              one Settings object, read from SWITCHBOARD_* env vars / .env
├── log.py                 structured JSON logging, configured explicitly by entry points
├── backends/
│   ├── base.py            Generation: text, token counts, latency, finish reason
│   └── local.py           vLLM client: chat, tokenize, served models, version; retries
└── labeling/
    ├── benchmarks.py      7 pinned benchmark sources → Items (router_text, user_prompt, reference)
    ├── splits.py          loads data/splits.yaml; rejects overlaps and unknown names
    ├── cache.py           SQLite generation cache keyed (model@revision, prompt, decode params)
    ├── sandbox.py         Docker sandbox for running model-written code
    ├── graders/           one grader per benchmark, all (prediction, reference) -> bool
    │   ├── common.py      "the answer is …", numbers, \boxed{} extraction
    │   ├── numeric.py     GSM8K
    │   ├── math.py        MATH
    │   ├── choice.py      MMLU, ARC
    │   ├── bbh.py         BIG-Bench Hard
    │   └── code.py        HumanEval, MBPP (via the sandbox)
    ├── generate.py        the label run: length check → generate → cache → grade → parquet
    ├── schema.py          required columns/dtypes of the labels file, checked on every write
    ├── selfcheck.py       oracle test of every grader against every real item
    ├── summary.py         labels → results/labels-summary.json + tables in reports/R1-labels.md
    └── review.py          30-item manual review sheet
data/splits.yaml           which benchmark is train / val / test
reports/R1-labels.md       Phase 1 report (tables generated, narrative written by hand)
tests/unit/                fast tests: no network, GPU or Docker
tests/integration/         tests against a live vLLM server
Makefile                   every sanctioned entry point
TODO.md                    checklist + decisions log
```

Not committed (regenerated by script, see `.gitignore`): `data/raw/` (downloaded datasets),
`data/cache/` (generations), `project-docs/` (private planning notes), secrets, Terraform state.

### 3.3 Where it runs

| Piece | Where | Why |
| --- | --- | --- |
| Code, tests, label runs | WSL2 Ubuntu, repo at `~/projects/switchboard` | vLLM targets Linux; WSL's own filesystem is fast and not synced by OneDrive |
| vLLM | WSL2, on the RTX 4060 (8 GB), port 8001 | Free GPU for the whole ML phase |
| Code sandbox | Docker Engine inside WSL | Model-written code must never run directly on the machine |
| Editing / commits | Windows copy in OneDrive, pushed to GitHub; WSL clone pulls | One source of truth for history |

---

## 4. Phase 1 — how labels are made

A **label** is a fact about the local model: *for this prompt, did it answer correctly?* Labels are
what the router learns from, so a wrong label poisons every later number. Most of Phase 1's effort
went into making labels trustworthy.

### 4.1 The flow

```
Hugging Face (pinned commit per dataset)
  └─▶ benchmarks.py: build Items, deterministic sample
        └─▶ for each Item not already in the cache:
              1. vLLM /tokenize: does the prompt + 1,024 answer tokens fit in 4,096?
                   no  → EXCLUDED (recorded with the reason; never generated or labelled)
                   yes → 2. vLLM chat completion (greedy, seed 0, max 1,024 tokens)
                          3. store in the SQLite cache (never overwritten)
        └─▶ grade every cached generation
              pure graders (maths, multiple choice, BBH) → in-process
              code graders (HumanEval, MBPP) → Docker sandbox, 4 in parallel
        └─▶ merge rows into data/labels/<model>.parquet (schema-checked)
        └─▶ update data/labels/<model>.meta.json (versions, settings, timings, exclusions)
  └─▶ summary.py → results/labels-summary.json + tables in reports/R1-labels.md
  └─▶ review.py  → reports/review/R1-sample.md (30 items for a human to check)
```

### 4.2 The benchmarks and the split

**The split is by dataset, never by row.** Whole benchmarks are held out. A random row split would
let the router learn *which benchmark a prompt came from* and predict that benchmark's average
accuracy — every metric would look excellent and none of it would generalise.

| Benchmark | Role | Items used | What it tests | Grader | Licence (per dataset card — verify before publishing) |
| --- | --- | ---: | --- | --- | --- |
| GSM8K | train | 3,000 (seeded sample of 8,792) | grade-school maths | numeric exact match | MIT |
| MMLU | train | 3,000 (seeded sample of 14,042) | broad knowledge, 4-option MC | option letter | MIT |
| MBPP | train | 974 (all) | short Python functions | unit tests in sandbox | CC-BY-4.0 |
| ARC-Challenge | val | 1,172 (test split) | science reasoning, MC | option letter | CC-BY-SA-4.0 |
| MATH-500 | test | 500 (all) | competition maths | `\boxed{}` + LaTeX normalisation | MIT (MATH) |
| HumanEval | test | 164 (all) | Python functions | unit tests in sandbox | MIT |
| BIG-Bench Hard | test | 1,620 (60 from each of 27 tasks) | multi-step reasoning | exact match by answer shape | MIT (BBH) |
| **Total** | | **10,430** | | | |

Every source is pinned to an exact Hugging Face commit hash, so the data can't change underneath
us. Sampling orders items by `sha256(seed:item_id)` and takes the first N — the same seed picks the
same rows on any machine and any library version.

### 4.3 Two texts per prompt

Each item carries:

- `router_text` — the task itself. **This is all the router will ever see.**
- `user_prompt` — the task plus a per-track answer-format instruction (e.g. *"Finish with a final
  line of the form 'The answer is: <number>'"*). **This is what the local model sees.**

The instructions differ by track. If the router saw them it could learn "this wording means GSM8K"
— provenance, not difficulty — and production prompts don't carry those instructions anyway.

### 4.4 Graders

All graders have the same shape — `(prediction, reference) -> bool` — and no network, model or
config access, so they are easy to test exhaustively.

| Grader | How it decides |
| --- | --- |
| GSM8K | Takes the number after the **last** "The answer is", else after `####`, else in `\boxed{}`, else the last number. Compares as exact decimals, so `42`, `42.0` and `The answer is 42` agree; `1,250.00` equals `1250`. |
| MATH | Takes the last `\boxed{…}` (with balanced braces) or "The answer is". Normalises LaTeX (`\dfrac`→`\frac`, strips `\left`, degrees, units, `x =`…), then compares strings, then numeric value (`0.5` = `\frac{1}{2}`). A bare number in prose is **not** accepted. |
| MMLU / ARC | Extracts an option letter, in order of precedence: an explicit marker ("The answer is (B)", "Answer: C", "the correct choice is C", "the best option is D"); a boxed letter (`\boxed{B}`); a **concluding option line** such as "**B. No, unless …**", accepted only when the line before it ends with a colon (so a bare list of options with no conclusion yields nothing); a lone "B" or "D. …" at the start. Only **uppercase** letters count, so "the answer is a mitochondrion" is not option A. |
| BBH | Looks at the target's shape: option `(C)` → compare letters; `True`/`No`/`invalid` → the first word if it is an answer word, otherwise the last answer word of that kind in the final clause ("The argument is valid.", "not valid" → invalid), and for yes/no a statement that someone lies or tells the truth reads as No/Yes; integer → compare numbers; bracket sequences → compare ignoring spaces; word lists → compare ignoring commas and case, order still required. A sentence with no answer word stays wrong. |
| HumanEval / MBPP | Pulls the code from the reply (prefers a fenced block containing `def`, handles a cut-off fence), builds a program with the benchmark's tests, and runs it in the sandbox. Pass = exit code 0. |

Empty answers and refusals grade as wrong everywhere.

### 4.5 The code sandbox

Model-written code is untrusted. Every program runs in a fresh container:

| Setting | Purpose |
| --- | --- |
| `--network none` | No internet access |
| `--memory 512m`, `--memory-swap 512m` | Can't exhaust RAM or swap |
| `--cpus 1`, `--pids-limit 64` | Can't hog CPU or fork-bomb |
| `--read-only` + 64 MB `/tmp` | Can't modify the container's filesystem |
| `--cap-drop ALL`, `no-new-privileges`, user `65534` (nobody) | No root, no privilege escalation |
| `timeout -s KILL 10` + outer timeout | Infinite loops are killed |
| Program sent over **stdin**, no volume mounts | No host folder is ever visible to the code |
| Image `python:3.11-slim@sha256:e41613d4…` | Pinned by digest, so grading can't silently change |

A separate `TrustedSubprocessSandbox` exists only for hand-written test fixtures, and the label run
refuses to use anything except the Docker sandbox for code benchmarks.

### 4.6 The generation cache

Label generation is the biggest GPU cost, so a generation is never produced twice.

- **Key** = `sha256(model_id@revision | sha256(messages) | sha256(decode params))`. Change the model
  checkpoint, the prompt text or the decoding settings and you get a new key automatically.
- **SQLite** in `data/cache/generations.sqlite`, committed every 50 items, so an interrupted run
  loses at most a few seconds of work.
- **`INSERT OR IGNORE`** — an existing generation is never overwritten.
- **Grading is separate from generation**: fixing a grader means re-grading from the cache in
  minutes, with zero GPU time.

### 4.7 Fixed decoding

Every benchmark uses one decode configuration: greedy (`temperature 0`), `top_p 1`, `seed 0`,
at most **1,024** new tokens. Greedy decoding means labels reflect the model, not sampling luck.
An answer cut off at 1,024 tokens is graded as it stands — failing to answer within the budget is
the model failing — and the summary reports how often that happened per benchmark.

### 4.8 Exclusions

A prompt whose length plus the 1,024-token answer budget exceeds vLLM's 4,096-token window cannot
be run under the fixed decode configuration. It is **excluded**: not generated, not labelled, not
counted as a failure, and recorded with its reason in `.meta.json` and in the summary. Measuring
all 10,430 prompts with the model's own tokenizer found exactly one: `mbpp-0493` (3,741 tokens).

### 4.9 Outputs

| File | Contents | Committed? |
| --- | --- | --- |
| `data/labels/Qwen__Qwen2.5-1.5B-Instruct.parquet` | One row per item: benchmark, id, prompt hash, router text, prompt, reference, generation, **label**, token counts, latency, finish reason, model + revision, decode hash | Yes (`5763eb7`) |
| `data/labels/…meta.json` | Model revision, decode params, vLLM version and settings, software versions, git commit, dataset revisions, per-run timings, exclusions, and every regrade with its before/after counts | Yes |
| `results/labels-summary.json` | Per-benchmark and overall accuracy, base rate, token-cap rate, flags | Yes |
| `reports/R1-labels.md` | Phase 1 report; tables between `GENERATED` markers are rewritten by the script | Yes |
| `reports/review/R1-sample-round1.md` | Review round 1 (seed 0): 2 mislabels, gate failed | Yes |
| `reports/review/R1-sample.md` | Review round 2 (fresh sample, seed 1): 1 mislabel, gate passed | Yes |

---

## 5. Every make command, explained

All commands run from `~/projects/switchboard` inside WSL. `make help` lists them.
(`make: *** No rule to make target …` means you are in a folder without the Makefile — `cd` first.)

### Development

| Command | What it does | When we ran it / result |
| --- | --- | --- |
| `make install` | `uv sync --frozen` builds `.venv` exactly from `uv.lock`, then installs the pre-commit git hooks | Once, after cloning into WSL |
| `make lint` | `ruff check` + `ruff format --check` on `src/` and `tests/` | Part of `make check` |
| `make format` | Applies ruff fixes and formatting | When lint complains |
| `make typecheck` | `mypy` in strict mode on `src/` | Part of `make check` |
| `make test` | Unit tests only — no network, GPU or Docker | Part of `make check`; currently **143 passed** |
| `make check` | `lint` + `typecheck` + `test` — what CI runs on every pull request | After every change |
| `make test-docker` | The 6 live sandbox tests: passing/failing programs, network blocked, infinite loop killed, filesystem read-only, end-to-end code grading | After installing Docker — **6 passed** |
| `make test-integration` | Against the running vLLM: the same prompt gives the same answer twice; the served model is the configured one | After starting vLLM — **3 passed** |

### Phase 1

| Command | What it does | When we ran it / result |
| --- | --- | --- |
| `make sandbox-pull` | Pulls `python:3.11-slim` **by digest** — the image the code graders run in | Once — pulled at the pinned digest |
| `make grader-selfcheck` | For every one of the 10,430 real items, builds an answer that must be right from the reference (or the benchmark's own canonical solution for code) and checks the grader accepts it; where easy, also checks a wrong answer is rejected | **0 failures on all 7 benchmarks** (about 6 minutes; code items run in Docker) |
| `make vllm` | Starts vLLM 0.30.0 serving the pinned Qwen checkpoint: bf16, 4,096-token context, up to 32 concurrent sequences, 85% of GPU memory, seed 0, port 8001. Runs in its own environment so its PyTorch/CUDA never clash with ours | Left running in terminal 1 during label runs |
| `make labels BENCH=<name>` | The label run for one benchmark (or `all`) — length check, generate what's missing, grade, write parquet + metadata. Safe to stop and rerun at any time | All 7 ✅ — 10,429 labelled, 1 excluded, 2,663,977 tokens in 0.48 h of generation |
| `make labels BENCH=<name> LIMIT=N` | Same, but only the first N items — a smoke test | `gsm8k LIMIT=20`, `mbpp LIMIT=10` |
| `make regrade BENCH=<name>` / `all` | Re-grades every cached generation with the current graders — no vLLM, no GPU — and records the before/after correct count in the metadata | After each grader fix: `all` (≈5 min, code runs in Docker), then `bbh` |
| `make labels-summary` | Reads the labels file and writes `results/labels-summary.json` + the tables in `reports/R1-labels.md` | After the smoke runs, the full run, and each regrade — overall base rate **60.0%** |
| `make review-sample` (`SEED=1` for a fresh draw) | Picks 30 random items (seeded) into a checklist to verify | Seed 0 → round 1 (failed, 2 mislabels); seed 1 → round 2 (passed, 1) |
| `make help` | Lists all targets | Any time |

---

## 6. Other commands we ran

| Command | Why |
| --- | --- |
| `git init -b main`, `.gitignore`, `.gitattributes` | Start the repo; ignore private docs, secrets, raw data, caches; force LF line endings because scripts run in Linux containers |
| `uv add --bounds exact …` | Add dependencies pinned to exact versions (no floating ranges) |
| `uv run pre-commit run --all-files` | Run gitleaks secret scanning, whitespace/YAML/TOML checks and ruff over the whole tree before committing |
| `git clone /mnt/c/…/LLMRouter ~/projects/switchboard` | Move the working copy into WSL's own filesystem (fast, not OneDrive-synced) |
| `curl -fsSL https://get.docker.com \| sh` + `usermod -aG docker` | Install Docker Engine in WSL for the sandbox |
| `curl localhost:8001/v1/models` | Check vLLM is up and serving the right model |
| Tokenizer measurement (one-off `uv run --with transformers`) | Counted every prompt's exact token length with the model's own chat template, which found the single over-length prompt |
| `git remote add origin https://github.com/MdJunaidAhmed16/SwitchBoard.git` + `git push` | Publish the history to GitHub |

---

## 7. How the project's principles are followed

### The seven non-negotiables

| Principle | How it is enforced today |
| --- | --- |
| **1. Public data only, machine-gradable, no LLM judges** | All 7 benchmarks are public and pinned. Every grader is deterministic code: exact match, option letters, or unit tests. Nothing asks a model whether an answer is right. |
| **2. One script reproduces every number** | Phase 1's numbers come only from `make labels` + `make labels-summary`; report tables are regenerated between markers and never typed. `make bench` will extend this to every published number. |
| **3. Split by dataset, never by row** | `data/splits.yaml` assigns whole benchmarks to train/val/test; the loader rejects a benchmark listed twice or an unknown name, and a test checks the committed split covers all 7 exactly once. |
| **4. Baselines ship with the result** | Built into Phase 2: heuristic and random use the same `predict_proba` interface as the router and are plotted on the same axes in the same run. |
| **5. Escalation is cheap, a wrong local answer is not** | Already applied: an answer cut off at the token limit counts as wrong; refusals and empty answers count as wrong; an over-length prompt is excluded, never guessed. The gateway's failure paths (Phase 4) all default to escalating. |
| **6. No fabricated numbers** | README figures are `TBD` until generated; reports use generated blocks; this file quotes measured results only from generated files. |
| **7. Cost ceiling enforced in code** | Frontier model and prices pinned in `config.py`; a hard per-run spend guard arrives with the frontier client in Phase 2; GPU budget alerts and idle shutdown in Phase 5. |

### Reproducibility

- Model by **commit hash**; datasets by **commit hash**; sandbox image by **digest**; every Python
  dependency at an **exact version** with `uv.lock` committed.
- Fixed seeds and greedy decoding.
- Every label run records software versions, the vLLM version and settings, and the git commit
  in `.meta.json`.
- The cache key includes model revision and decode settings, so stale generations can't be reused.

### Testing — effort where bugs hide

The docs single out graders as the most dangerous component: a grader bug silently corrupts every
label and number and is nearly invisible by inspection. So:

- **12–21 hand-written fixtures per grader**, including adversarial ones (extra prose, wrong
  format, empty output, refusals, lowercase "a", pronoun "I", a minus sign).
- **Grader self-check over all 10,430 real items** — checks graders against the real reference
  formats, not just fixtures.
- **Every "wrong" answer in the smoke runs was read by hand** to confirm the grader, not the
  model, wasn't the one making the mistake.
- Pipeline tests drive generate → cache → grade → parquet → summary against a fake vLLM server,
  including resume-without-regenerate and exclusions.
- Live tests for the sandbox (Docker) and determinism (vLLM), kept separate so `make test` stays
  fast and runs anywhere.

### Engineering conventions

- **One config object** (`config.py`); no other module reads environment variables.
- **Structured JSON logs** via structlog; no `print` in `src/`.
- **Strict typing** (`mypy --strict`), `ruff`, line length 100.
- **Conventional Commits** with scopes (`feat(labeling): …`, `fix(labeling): …`); one logical change
  per commit; bodies explain what and why.
- **Short-lived branches** named `<type>/<scope>-<slug>` (`feat/labeling-pipeline`), merged into
  `main` once the phase's gate passes.
- **Decisions are written down** in `TODO.md`'s decisions log — a decision that only exists in a
  chat doesn't exist.

### Security

- `.env`, keys, credentials, Terraform state and raw data are git-ignored.
- **gitleaks** scans for secrets on every commit (pre-commit) and in CI.
- Model-written code only runs in the locked-down sandbox.
- The project does not interpret prompts as instructions; prompt-injection risk is inherited from
  the serving model, not introduced by the router, and the README says so rather than claiming more.

---

## 8. Problems we hit and how they were fixed

| Problem | Cause | Fix |
| --- | --- | --- |
| Grader picked the *first* "answer is" on a line | A greedy regex swallowed later markers on the same line | Anchor on the last marker; caught by a fixture before any real run |
| `The answer is -7` graded as 7 | The minus sign was read as a separator after "is" | A dash only counts as a separator when not followed by a digit; caught by a fixture |
| BBH download failed midway | Transient Hugging Face disconnect across 27 files | Use the local copy first, retry downloads with backoff |
| `make: *** No rule to make target 'sandbox-pull'` | Command run from the home folder after restarting WSL | Always `cd ~/projects/switchboard` first (now in every instruction) |
| Smoke runs didn't record vLLM settings | Smoke runs called the script directly, bypassing the Makefile | Added `LIMIT=N` to `make labels` |
| Thousands of `HTTP Request: POST …` log lines | httpx logs every request at INFO | httpx/httpcore set to WARNING; warnings and errors still show |
| `make labels BENCH=mbpp` → Error 1 on `mbpp-0493` | 3,741-token prompt + 1,024 answer tokens > 4,096 context | Check length via vLLM `/tokenize` before generating; exclude and record over-length prompts |
| **Review round 1 failed: 2 of 30 mislabelled** | Correct answers in non-template forms graded wrong: a bolded concluding option line (MMLU) and "Jim does not tell the truth" for a Yes/No question (BBH). A scan showed both were systematic: no letter from 108 of 4,172 MC replies; 113 of 208 wrong BBH yes/no-type answers didn't start with the answer word | Graders extended (boxed letters, "correct choice is", concluding option lines, answer words anywhere in the final clause, truth-teller statements, bold "Final Answer"); every label change listed before applying: 94 wrong→correct, **0 correct→wrong**; `make regrade` added; fresh round 2 drawn |
| Review round 2: 1 of 30 mislabelled (gate passed) | A correctly sorted word list written with commas and capitals | Word-list answers compared ignoring commas and case; the complete set of affected labels (4) checked |

---

## 9. Decisions and why

The full table with dates is in `TODO.md` → *Decisions log*. In short:

| Decision | Reason |
| --- | --- |
| Local model Qwen2.5-1.5B-Instruct | A 3B model's weights (~6.2 GB) leave almost no KV cache on an 8 GB GPU; 1.5B is Apache-2.0 |
| Frontier model Claude Opus 5.5 | Owner's choice of provider; current default Claude model; prices pinned in config |
| Call it through OpenRouter | Owner's key provider; OpenAI-compatible, so it shares the vLLM client's shape; per-key credit limit = extra spend ceiling |
| Fast-forward merges, not squash | Keeps every commit and its real date on `main`, so history and the contribution graph reflect the work; nothing is backdated |
| ARC-Challenge as validation | Validation must be a held-out *benchmark*; ARC is public, multiple choice, and in neither train nor test |
| MATH-500 for MATH | The original MATH dataset repo was taken down; MATH-500 is the standard subset |
| Keep HumanEval at 164 items | Below the "several hundred" guideline, so it's reported with confidence intervals |
| Router sees task text only | Format instructions would leak which benchmark a prompt came from |
| One decode config, 1,024 tokens | One cache hash; hitting the cap counts as failing |
| Exclude prompts that can't fit | Smaller budget for one item would break the single decode config |
| Grade non-template answer forms | A strict template undercounted correct answers and would have biased the router toward escalating; every resulting change was audited |
| Fresh sample for review round 2 | Re-checking the items a fix was designed around would not test the fix |
| vLLM as an external server | Its pinned PyTorch/CUDA stays out of the project's environment |
| One container per program, code over stdin | Nothing on the host is visible to model-written code |

---

## 10. Git history

Branches: `main` (always deployable) and `feat/labeling-pipeline` (Phase 1, merged after the
gate). Remote: `https://github.com/MdJunaidAhmed16/SwitchBoard`.

| Commit | Date | Change |
| --- | --- | --- |
| `1806f6c` | 2026-09-30 | build: scaffold repository, tooling, config and CI |
| `8a67004` | 2026-09-30 | feat(labeling): add benchmark graders and code-execution sandbox |
| `7bb3ee0` | 2026-09-30 | feat(labeling): add pinned benchmark registry and dataset-level split |
| `80c2a32` | 2026-09-30 | feat(labeling): add cached label generation, summary and review tools |
| `6587faa` | 2026-09-30 | test(labeling): add grader self-check over every real benchmark item |
| `d487fc1` | 2026-09-30 | docs: mark Phase 1 code and tests complete in the tracker |
| `a5b3db5` | 2026-10-02 | feat(backends): pin the frontier model and its prices in config |
| `5aedf72` | 2026-10-02 | docs: record owner decisions and Docker verification in the tracker |
| `61ace51` | 2026-10-02 | build(labeling): add LIMIT to make labels for smoke runs |
| `8f13458` | 2026-10-02 | fix(labeling): stop httpx logging every vLLM request at INFO |
| `6145912` | 2026-10-02 | docs: record live vLLM checks and smoke runs in the tracker |
| `1fe2c8c` | 2026-10-02 | docs: record paused label run and the over-length MBPP prompt |
| `5c66c33` | 2026-10-02 | fix(labeling): exclude prompts that cannot fit the context window |
| `ddb5391` | 2026-10-02 | docs: record the context-length exclusion fix in the tracker |
| `4aa5caa` | 2026-10-02 | docs: add whatsDone.md, a running explanation of the build |
| `0d43f83` | 2026-10-02 | docs: log the GitHub push and the first CI run in whatsDone |
| `03ebccd` | 2026-10-02 | fix(labeling): accept correct answers given in non-template forms |
| `d97a56d` | 2026-10-02 | feat(labeling): add make regrade to re-grade labels from the cache |
| `5a9f100` | 2026-10-02 | fix(labeling): ignore separators and case in BBH word-list answers |
| `5763eb7` | 2026-10-02 | bench(labeling): Phase 1 labels for Qwen2.5-1.5B-Instruct, gate passed |
| `c11e806` | 2026-10-02 | docs: record Phase 1 completion in whatsDone and the tracker |

---

## 11. Progress log

Newest entries at the bottom. Each entry: what was done, how it was verified, what's next.

### 2026-09-30 — Stage 0: foundation
- Repo, `.gitignore` (private docs, secrets, raw data, caches), LF line endings.
- Python 3.11 project with exact-pinned dependencies and `uv.lock`; ruff, strict mypy, pytest.
- Single `Settings` config; structured JSON logging; gitleaks pre-commit; GitHub Actions CI.
- README skeleton with the claim, both baselines and both failure conditions; every figure `TBD`.
- **Verified:** lint, types and tests clean; pre-commit hooks pass on every file.

### 2026-09-30 — Phase 1 code
- Graders for all 7 benchmarks with adversarial fixtures; Docker sandbox; pinned benchmark
  registry with dataset-level split; generation cache; vLLM client; label run, summary, review
  sheet; grader self-check.
- **Verified:** 141 unit tests; self-check 0 failures on all 9,292 non-code items; all 1,138
  canonical code solutions pass assembly. Fixtures caught two real grader bugs (§8).

### 2026-10-02 — Owner decisions and machine setup
- Confirmed Qwen2.5-1.5B-Instruct; chose Claude Opus 5.5 as frontier (pinned in config); kept
  HumanEval.
- Repo moved into WSL; Docker installed.
- **Verified:** `make test-docker` 6 passed; `make grader-selfcheck` 0 failures on all 10,430
  items with code running in Docker; vLLM 0.30.0 up; `make test-integration` 3 passed.

### 2026-10-02 — Smoke runs and fixes
- `make labels BENCH=gsm8k LIMIT=20` and `BENCH=mbpp LIMIT=10`: every item labelled, none failed,
  none hit the token cap. Every item graded wrong was read by hand: GSM8K misses were wrong final
  answers; MBPP misses were genuine `AssertionError`s in the sandbox, not harness errors.
- Added `LIMIT=N` to `make labels`; silenced per-request HTTP logging.

### 2026-10-02 — Full label run, part 1, and the over-length prompt
- `gsm8k` and `mmlu` labelled in full. `mbpp` stopped with Error 1 on `mbpp-0493` (prompt too long
  for the context window).
- Measured all 10,430 prompts with the model's tokenizer: `mbpp-0493` is the only one that can't
  fit. Added the pre-generation length check and recorded exclusions.
- **Verified:** 143 unit tests, including two new exclusion tests.
- GitHub remote added; `main` and `feat/labeling-pipeline` pushed, history identical to local.
  Before pushing, history was checked: every author/committer is the owner, no co-author
  trailers, no secret patterns.
- **CI on `main`:** lint, type check, unit tests and the gitleaks scan of the full history all
  pass. The Docker step fails only because `main` is still the Stage 0 commit, where no Docker
  tests existed yet and pytest reports "no tests collected" as an error. The feature branch has
  the 6 Docker tests and passes this step locally; `main` goes green when Phase 1 is merged.
- **Next:** rerun `mbpp`, finish `arc_challenge` (running now), `math`, `humaneval`, `bbh`; then
  `make labels-summary`, the 30-item review gate, and the R1 report.

### 2026-10-02 — Phase 1 complete: labels, review gate, R1 report
- **Full label run finished:** all 7 benchmarks, 10,429 items labelled, 1 excluded (`mbpp-0493`),
  2,663,977 tokens generated in 0.48 h of generation wall-clock on the laptop GPU.
- **Review round 1 (seed 0) failed — 2 of 30 mislabelled.** Both were correct answers the grader
  missed because they weren't in the requested template. Scanning all labels showed the patterns
  were systematic, so graders were fixed with new fixtures (including negative cases), every
  label change was listed and checked before applying (94 wrong→correct, 0 correct→wrong; all 30
  multiple-choice flips from the new concluding-line rule read individually), and all labels were
  re-graded from the cache with the new `make regrade` — no GPU needed.
- **Review round 2 (fresh sample, seed 1) passed — 1 of 30.** The one miss (a correctly sorted
  word list with commas) was also fixed; its complete set of 4 affected labels was checked.
- **Results** (from `results/labels-summary.json`): overall base rate **60.0%** (6,259 / 10,429);
  train 64.7%, validation 68.8%, test 41.2%. Per-benchmark accuracy and token-cap rates are in
  `reports/R1-labels.md`, which now has its setup, review history, surprises and consequences
  written up.
- **Verified:** 177 unit tests pass; the grader self-check still passes on all 10,430 items.
- **Open:** owner spot-check of both review sheets (the review was done by an AI assistant at the
  owner's request); merge `feat/labeling-pipeline` into `main`; licence verification at source.
- **Next:** Phase 2 — split-integrity test first, then the frontier client with a spend guard,
  the heuristic and random baselines, the v0 router, the cost model and threshold sweep, and the
  kill gate.

### 2026-10-03 — Phase 1 merged; frontier provider set to OpenRouter
- **Why the GitHub graph showed only 1 contribution:** GitHub counts commits only once they are on
  the default branch (`main`). All Phase 1 work was on `feat/labeling-pipeline`, so the only
  contribution was the repository's creation.
- **Merged** `feat/labeling-pipeline` into `main` by **fast-forward**: all 21 commits are now on
  `main` with their original dates (6 on 2026-09-30, 15 on 2026-10-02). CI on `main` is green.
  From now on each step is merged into `main` the same day it is finished, so the graph follows
  the actual work. Commits are never backdated.
- **Frontier provider → OpenRouter.** `config.py` now points at `https://openrouter.ai/api/v1`
  with model `anthropic/claude-opus-5.5` ($4 / $20 per 1M listed on OpenRouter, checked
  2026-10-03). The key is read from `OPENROUTER_API_KEY` in the gitignored `.env` as a secret
  value that never appears in logs or reprs (tested).
- **Verified:** 179 unit tests; gitleaks clean.
- **Next:** Phase 2, starting with the split-integrity test.

### 2026-10-03 — Pull-request workflow
- **Why there were no PRs:** Phase 1 and the OpenRouter change were merged locally and pushed
  straight to `main`, which bypasses GitHub's pull-request flow. Those merges are already in
  `main`, so they can't become PRs after the fact.
- **From now on every step is a PR:** work happens on a `<type>/<scope>-<slug>` branch, the branch
  is pushed, a PR is opened with what changed / how it was verified / what to test (template in
  `.github/pull_request_template.md`), CI runs on the PR, and it is merged with a **merge
  commit** (not squash), so every commit keeps its real date.

### 2026-10-03 — Phase 2, part 1: routers, and the kill gate fails
- **Built:** the split-integrity guard (runs on the committed labels, not skippable); one `Router`
  interface; the **heuristic** (log word count + keyword count → logistic regression, keyword list
  fixed in advance); the **random** router; **v0** (frozen `BAAI/bge-base-en-v1.5` embeddings,
  cached, + logistic regression with C chosen on validation); metrics with bootstrap intervals;
  and `make train-v0`, which writes `results/router-v0.json`, per-prompt scores and the generated
  R2 tables. Dependencies added: PyTorch (CUDA), transformers, scikit-learn, matplotlib.
- **Result** (from `results/router-v0.json`): on the held-out test benchmarks the heuristic scores
  AUROC 0.606, v0 0.549, random 0.507. v0 is significantly *below* the heuristic (paired interval
  entirely under zero), and its interval's lower end (0.524) misses the pre-registered 0.55 floor:
  **kill-gate criterion 3 fails.**
- **Why:** v0 learned which benchmark a prompt came from. Its in-sample training AUROC is 0.755
  and its mean score per training benchmark tracks the base rates; on a random row split it would
  have looked strong (0.712), which is precisely the inflation the dataset-level split exists to
  expose.
- **What happens now:** per the roadmap, building stops (no serving work) until the owner chooses:
  write up the negative result, change the benchmark mix / re-weight training, or change the local
  model. The frontier answers are needed for the cost curve under every option.
- **Verified:** 201 unit tests; results identical across two runs.

### 2026-10-05 → 2026-10-07 — Attempt 2, fine-tuned v1, and the frontier pilot
- **Owner's decision:** a second attempt (wider training mix + re-weighting) and evaluate the
  fine-tuned router as well. The design was **pre-registered in R2 before any result existed**.
- **Six training benchmarks added** (GSM-Hard, SVAMP, AQuA-RAT, CommonsenseQA, MedMCQA, QASC):
  8,016 new labels on the laptop GPU, none failed. GSM-Hard scores 40.7% against GSM8K's 80.9% on
  prompts that read alike — the contradiction needed to break the shortcut. A scan for unparsable
  answers found mostly genuine non-answers; answer-by-text misses are at most 0.06% of labels.
- **New code:** attempts defined in code (attempt 1 still reproduces exactly); benchmark-balanced
  training weights; **v1**, the same encoder fine-tuned end to end over 3 seeds; temperature scaling
  on validation; an OpenRouter frontier client whose spend guard refuses a request *before* sending
  it; `make train-attempt2`, `make frontier-pilot`, `make frontier MAX_USD=…`.
- **Problem and fix:** the first attempt-2 run was killed with the session after 50 minutes and
  lost all progress. Training now batches prompts of similar length (several times faster: ~17
  minutes per seed), checkpoints each finished seed, logs progress every 100 steps, and runs
  detached so it survives the session.
- **Results** (from `results/router-attempt2.json`): heuristic test AUROC 0.607; v0 0.564 (up from
  0.549, still failing criterion 3); **v1 0.596 mean over three seeds**, passing criterion 3 on every
  seed and beating v0 by more than its seed spread — but **tied with the heuristic** (every paired
  interval includes zero). v1 leads on MATH, trails on HumanEval. Calibration improves on every seed
  but test ECE stays above the 0.05 target.
- **Frontier pilot:** Claude Opus 5.5 answered 30 test prompts for $0.23; the cost computed from
  tokens × pinned prices matched OpenRouter's charge exactly. Projected cost for all 2,284 test
  prompts: about $14.75.
- **Verified:** 224 unit tests; attempt-1 numbers reproduce exactly after the refactor.
- **Next:** with the owner's approval, the full frontier run, then the cost model, the threshold
  sweep and the cost-quality curve — which decide whether v1's tie on AUROC is a saving in money.
