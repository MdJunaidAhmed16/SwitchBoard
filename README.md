# Switchboard

Switchboard tests whether a small classifier can route each prompt to the cheapest model that can
still answer it, by measuring the saving as a cost-versus-quality curve on public benchmarks.

> **Status: complete, as a negative result.** The pre-registered kill gate failed: on held-out
> benchmarks no learned router beat a two-feature heuristic, and the money saved at 95% of
> frontier quality was under 10%. Serving infrastructure was deliberately not built. Every number
> below is generated from the committed results files by `make readme`; none is typed by hand.

## Headline

<!-- BEGIN GENERATED: readme-headline -->
> At 95% of always-frontier quality, the learned router (v1, three seeds) keeps **9.0%–10.7%** of traffic on the local model, at **90.4%–91.3%** of always-frontier cost. The length-and-keyword heuristic keeps 11.4% at 91.8%; random routing 7.9% at 93.0%.
>
> No learned router's cost differs significantly from the heuristic's (every 95% bootstrap interval includes zero). **The pre-registered kill gate failed**: no router beats both the heuristic and random routing on any seed.
<!-- END GENERATED: readme-headline -->

```bash
make sweep && make readme   # regenerates the curve, the gate verdict and every number here
```

![Cost-quality curve](reports/figures/cost-quality-attempt2.png)

Every router — learned, heuristic or random — sits close to the straight line between
always-local and always-frontier. That line is what you get by routing a share of traffic at
random, so a curve on it means the router found almost nothing worth keeping local.

## What was found

1. **The local model's accuracy caps the saving, whatever the router does.** The 1.5B local model
   is right on 40% of the test prompts and Claude Opus 5.5 on 98%. Holding 95% of frontier quality
   leaves room to keep only about a tenth of traffic local, so even a good router saves little.
2. **A naive learned router learns where a prompt came from, not how hard it is.** Trained on three
   benchmarks, the frozen-encoder router scored 0.71 AUROC on a random row split and 0.55 on unseen
   benchmarks — it had learned each benchmark's base rate. A wider, re-weighted training mix
   (attempt 2) removed that shortcut.
3. **Fine-tuning helped, but only up to the heuristic.** The fine-tuned encoder (v1) beat the
   frozen one beyond its seed spread, and tied a logistic regression on prompt length and a
   twelve-word keyword list. Temperature scaling improved calibration on every seed, but not to
   the 0.05 ECE target.
4. **AUROC gains of a few points did not turn into money.** At these base rates the cost curves of
   all routers lie within the bootstrap noise of one another.

The full analysis, including both pre-registrations, is in
[`reports/R2-kill-gate.md`](reports/R2-kill-gate.md); the label pipeline and its audits are in
[`reports/R1-labels.md`](reports/R1-labels.md).

## The claim, and how it was allowed to fail

A small classifier can predict, from the prompt alone and before any generation, whether a cheap
self-hosted model will answer correctly. Routing on that prediction should keep most of the quality
of always-frontier at a fraction of the cost.

Written before any training code:

1. **If the heuristic baseline's curve matches or dominates the learned router's curve, the model is
   unnecessary.** — *This happened for v0 and two of the three v1 seeds; the third beat the heuristic
   by margins its bootstrap intervals cannot separate from zero.*
2. **If random routing at the same escalation rate matches the router's quality, the classifier has
   not learned a usable signal.** — *This happened for v0 and two of the three v1 seeds, at one or more
   escalation rates.* No router passed both.

The roadmap's rule was to stop rather than build serving around a result that does not exist.
The exact pass/fail definitions were committed before the curves were computed.

## Results

<!-- BEGIN GENERATED: readme-results -->
**Base rates** — how often each model is already right.

| | Local model (Qwen2.5-1.5B) | Frontier (Claude Opus 5.5) |
| --- | --- | --- |
| Train benchmarks (14,989 prompts) | 59.2% | — |
| Test benchmarks (2,284 prompts) | 41.2% | — |
| Test subset with frontier answers (457 prompts) | 40.0% | 97.8% |
| Cost per 1,000 prompts on the subset | $0.08 | $7.58 |

**Routers** — AUROC on all 2,284 test prompts (95% bootstrap interval); cost and gate criteria on the 457-prompt subset at 95% quality retention.

| Router | Test AUROC | ECE | Kept local | Cost vs always-frontier | C1: beats heuristic | C2: beats random |
| --- | --- | --- | --- | --- | --- | --- |
| heuristic | 0.607 [0.582, 0.629] | 0.074 | 11.4% | 91.8% | — | — |
| random | 0.507 [0.485, 0.532] | 0.248 | 7.9% | 93.0% | — | — |
| v0 | 0.564 [0.542, 0.586] | 0.075 | 7.0% | 93.9% | **fail** | **fail** |
| v1_s0 | 0.596 [0.572, 0.620] | 0.149 → 0.095 | 9.0% | 91.3% | **fail** | pass |
| v1_s1 | 0.604 [0.579, 0.628] | 0.131 → 0.072 | 10.7% | 90.4% | pass | **fail** |
| v1_s2 | 0.587 [0.563, 0.611] | 0.192 → 0.080 | 10.5% | 90.5% | **fail** | **fail** |

v1 mean test AUROC 0.596 (seed spread 0.017); ECE for v1 is before → after temperature scaling.

**Why the split matters** — the same frozen-encoder router (v0), scored two ways.

| | Dataset-level split (test benchmarks never seen) | Row split — optimistic |
| --- | --- | --- |
| Attempt 1 (train: GSM8K, MMLU, MBPP) | 0.549 [0.524, 0.572] | 0.712 [0.690, 0.734] |
| Attempt 2 (nine train benchmarks, balanced weights) | 0.564 [0.542, 0.586] | 0.599 [0.580, 0.617] |

Charts: `reports/figures/cost-quality-attempt2.png`, `reports/figures/reliability-attempt2.png`.
<!-- END GENERATED: readme-results -->

![Reliability diagram](reports/figures/reliability-attempt2.png)

The heuristic and v0 rarely leave the 0.35–0.65 band: they almost never make a confident call. v1
spreads its scores wider, which is where its AUROC gain over v0 comes from, but even after
temperature scaling it is over-confident above 0.6.

## How it works

1. **Labels.** The local model (Qwen2.5-1.5B-Instruct, served by vLLM on a laptop RTX 4060) answers
   every prompt from 13 public benchmarks once, greedily. Each answer is graded by machine: exact
   match for numbers, the chosen letter for multiple choice, unit tests in a network-less Docker
   sandbox for code. That gives 18,445 (prompt, local model was right) pairs.
2. **Routers.** Each router reads only the prompt and outputs the probability that the local model
   will be right: a length-and-keyword logistic regression (the heuristic), a hash-based random
   score, a frozen `bge-base-en-v1.5` encoder with logistic regression (v0), and the same encoder
   fine-tuned end to end with temperature scaling (v1, three seeds).
3. **Frontier answers.** Claude Opus 5.5, through OpenRouter, answers a stratified 20% of the test
   prompts (457), within a $4 budget enforced by a spend guard that refuses a request before
   sending it if the worst case could exceed the cap.
4. **Sweep.** For every threshold τ from 0 to 1, prompts scoring at least τ stay local and the rest
   escalate. Quality and cost are pure arithmetic over the cached answers; cost is tokens × pinned
   list prices.

## Evaluation design

- **Split by dataset, never by row.** Train: GSM8K, MMLU, MBPP, GSM-Hard, SVAMP, AQuA-RAT,
  CommonsenseQA, MedMCQA, QASC. Validation: ARC-Challenge. Test: MATH, HumanEval, BIG-Bench Hard.
  (`data/splits.yaml`, with a test that fails if any prompt crosses splits.)
- A row-split figure is reported beside it, labelled **optimistic**, with the gap explained.
- Every accuracy is stated next to its **base rate**.
- Only machine-gradable benchmarks. No LLM-as-judge, no human grading; 30-item manual audits of
  the grader were run and recorded in `reports/review/`.
- Thresholds, calibration and the regularisation strength are chosen on validation only; the test
  benchmarks are touched once per pre-registered run.
- Model revisions, prompts and decoding parameters are pinned, and every generation is cached by
  (model@revision, prompt, decoding parameters), so re-running a step never re-queries a model.

## Reproducing the result

```bash
make install                         # env from uv.lock (Python 3.11)
make sandbox-pull                    # digest-pinned Docker image for the code graders
make vllm                            # terminal 1: serve the local model on the GPU
make labels BENCH=all                # terminal 2: generate + grade every benchmark
make train-v0                        # attempt 1 (frozen encoder, narrow mix)
make train-attempt2                  # attempt 2: heuristic, random, v0, v1 x 3 seeds
make frontier FRACTION=0.2 MAX_USD=4 # Claude answers for the test subset (needs OPENROUTER_API_KEY)
make sweep                           # cost-quality curve and gate criteria 1-2
make readme                          # regenerate this README's numbers and the reliability diagram
make check                           # lint, strict typecheck, unit tests
```

The committed `data/labels/` and `results/` files let `make sweep` and `make readme` run without a
GPU or an API key. Re-asking Claude for the subset costs about $3.50 at list prices.

## Limitations

- **One local model.** Labels are specific to Qwen2.5-1.5B. A stronger local model raises the
  ceiling on savings, and this result says nothing about how a router would do then.
- **A 457-prompt subset decides the cost curve**, for budget reasons; its intervals are wide. AUROC
  uses all 2,284 test prompts.
- Benchmark prompts are short and clean; production traffic is not.
- Prompt-only routing is strictly weaker than methods that inspect the generation.
- Local cost is a market price for serving a small open model, not a measured GPU cost; the
  verdict is unchanged at zero and at five times that price.
- Single-turn only. Multi-turn routing is a different problem.
- The dataset-level split is harsh; the row-split figure is the optimistic one. Both are reported.

## Related work

- **Cascades** (e.g. FrugalGPT) call the cheap model first and escalate after scoring its answer;
  they pay for a local generation on every prompt, which prompt-only routing avoids.
- **Generation-aware routing** (e.g. AutoMix's self-verification) judges the cheap model's answer
  before escalating; stronger signal, higher latency.
- **Learned prompt-only routers** (e.g. RouteLLM, Hybrid LLM) train on preference or quality-gap
  data and report sizeable savings, mostly with random or in-distribution splits; this project
  tests the same idea on benchmarks the router has never seen and against a heuristic baseline.

## Security posture

This system does not interpret prompts, so injection risk is inherited from the serving model
rather than introduced by the router. Code from model answers runs only in a network-less,
read-only, resource-limited Docker sandbox. API keys live in a gitignored `.env` and are held as
secrets in memory; no key is ever logged.
