# Switchboard

Switchboard routes each prompt to the cheapest model that can still answer it correctly, and
measures the saving as a cost-versus-quality curve on public benchmarks.

> **Status: Phase 1 (labels) in progress. No results yet.** Every figure below is `TBD` until it is
> generated from `results/latest.json` by `make bench`. No number in this README is typed by hand.

## Headline

> At τ = `TBD`, Switchboard keeps **`TBD`%** of traffic on the local model, retains **`TBD`%** of
> always-frontier quality, at **`TBD`%** of the cost.

```bash
make bench   # reproduces every number in this README
```

## The claim

A small classifier can predict, from the prompt alone and before any generation, whether a cheap
self-hosted model will answer correctly. Routing on that prediction should keep most of the quality
of always-frontier at a fraction of the cost.

The deliverable is a **curve, not a number**: for every achievable quality level, what does the
traffic cost?

## Baselines

Both are computed in the same run and plotted on the same axes as the learned router.

- **Always-frontier.** Every prompt goes to the frontier model. The quality and cost ceiling.
- **Length-and-keyword heuristic.** Prompt token count plus a small keyword list, fitted with
  logistic regression. What a sceptic says could be built in an afternoon without ML.

A third reference, **random routing at the same escalation rate**, is plotted for diagnosis.

## How this project can fail

Written before any training code, and reported against honestly.

1. **If the heuristic baseline's curve matches or dominates the learned router's curve, the model is
   unnecessary.** That is reported as a negative result.
2. **If random routing at the same escalation rate matches the router's quality, the classifier has
   not learned a usable signal.**

If either happens, this README's headline becomes that finding: *when does prompt-only routing
fail, and why*.

## Evaluation design

- **Split by dataset, never by row.** Train: GSM8K, MMLU, MBPP. Validation: ARC-Challenge.
  Test: MATH, HumanEval, BIG-Bench Hard. (`data/splits.yaml`)
- A row-split figure is also reported, labelled **optimistic**, with the gap explained.
- Every accuracy is stated next to the **base rate** — the fraction of prompts the local model
  already gets right.
- Only machine-gradable public benchmarks: exact match, multiple choice, or unit-test execution.
  No LLM-as-judge, no human grading.

## Results

| Metric | Value |
| --- | --- |
| Local model base rate (overall) | `TBD` |
| Router AUROC, dataset-level split (v0 / v1) | `TBD` / `TBD` |
| Router AUROC, row split — optimistic | `TBD` |
| ECE after calibration | `TBD` |
| Router p99 latency | `TBD` |

Cost-quality curve: `TBD`. Reliability diagram: `TBD`.

## Limitations

- Benchmark prompts are short and clean; production traffic is not.
- Labels are specific to one local model. Changing that model invalidates them.
- Prompt-only routing is strictly weaker than methods that inspect the generation.
- Local cost assumes a saturated GPU; at low utilisation the API is cheaper per request.
- Single-turn only. Multi-turn routing is a different problem.
- The dataset-level split is harsh; the row-split figure is the optimistic one. Both are reported.

## Security posture

This system does not interpret prompts, so injection risk is inherited from the serving model
rather than introduced by the router. It is a public demo with sensible limits, not a hardened
production service.

## Related work

`TBD` — cascades, generation-aware routing, published routers, one line each on how they differ.
