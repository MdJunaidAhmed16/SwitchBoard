"""Router evaluation on the dataset-level split, one pre-registered attempt at a time.

    python -m switchboard.router.evaluate --attempt 1     (make train-v0)
    python -m switchboard.router.evaluate --attempt 2     (make train-attempt2)

Attempt 1: heuristic, random and v0 trained on GSM8K, MMLU and MBPP. Attempt 2: the wider training
mix with benchmark-balanced weights, plus v1 fine-tuned over three seeds (see ``attempts.py``).

Writes ``results/<stem>.json`` (every number), ``results/<stem>-scores.parquet`` (one score per
router per labelled prompt, for the cost-quality sweep) and that attempt's generated block in
``reports/R2-kill-gate.md``. The stem is ``router-v0`` for attempt 1, ``router-attempt2`` after.

What this run can and cannot decide. Kill-gate criterion 3 (AUROC meaningfully above 0.5) is
evaluated here. Criteria 1 and 2 (v0's cost-quality curve dominates the heuristic's and beats random
routing at matched escalation) need the frontier model's answers and the cost model, and are
evaluated by the sweep. The AUROC comparisons below are diagnostics for those, not substitutes.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd

from switchboard.config import Settings, get_settings
from switchboard.labeling.generate import _git_commit
from switchboard.labeling.summary import write_into_report
from switchboard.log import configure_logging, get_logger
from switchboard.router.attempts import ATTEMPTS, V1_SEEDS, Attempt, balanced_weights
from switchboard.router.baselines import KEYWORDS, HeuristicRouter, RandomRouter
from switchboard.router.data import RouterData, load_router_data, row_split
from switchboard.router.encoder import FrozenEncoder
from switchboard.router.interface import Router
from switchboard.router.metrics import (
    auroc,
    base_rate,
    bootstrap_ci,
    brier,
    ece,
    paired_auroc_difference_ci,
)
from switchboard.router.v0 import V0Router
from switchboard.router.v1 import V1Router, pretrained_backbone

log = get_logger(__name__)

# Kill-gate criterion 3, fixed before the first evaluation run: the lower end of v0's 95%
# bootstrap interval for test AUROC must clear this, not merely 0.5.
AUROC_FLOOR = 0.55


def _split_metrics(y: np.ndarray, p: np.ndarray, seed: int) -> dict[str, Any]:
    lo, hi = bootstrap_ci(y, p, seed=seed)
    return {
        "n": len(y),
        "base_rate": base_rate(y),
        "auroc": auroc(y, p),
        "auroc_ci95": [lo, hi],
        "ece": ece(y, p),
        "brier": brier(y, p),
    }


def _per_benchmark(frame: pd.DataFrame, p: np.ndarray, seed: int) -> dict[str, Any]:
    out = {}
    for name, idx in frame.groupby("benchmark").indices.items():
        y = frame["label"].to_numpy(dtype=int)[idx]
        lo, hi = bootstrap_ci(y, p[idx], seed=seed)
        out[str(name)] = {"n": len(idx), "base_rate": base_rate(y), "auroc": auroc(y, p[idx]),
                          "auroc_ci95": [lo, hi]}  # fmt: skip
    return out


def _row_split(
    data: RouterData, v0_c: float, encoder: FrozenEncoder, seed: int, balanced: bool = False
) -> dict[str, Any]:
    """Optimistic comparison: random 80/20 split over all rows, same model settings."""
    train, test = row_split(data.all().drop(columns="weight", errors="ignore"), seed=seed)
    if balanced:
        train = train.assign(weight=balanced_weights(train).to_numpy())
    heuristic = HeuristicRouter()
    heuristic.fit(train, train)
    v0 = V0Router(encoder, seed=seed)
    v0.fit_fixed(train, v0_c)
    y = test["label"].to_numpy(dtype=int)
    out: dict[str, Any] = {"n_train": len(train), "n_test": len(test)}
    for router in (heuristic, RandomRouter(seed), v0):
        p = router.predict_proba(list(test["router_text"]))
        lo, hi = bootstrap_ci(y, p, seed=seed)
        out[router.name] = {"auroc": auroc(y, p), "auroc_ci95": [lo, hi]}
    return out


def _diagnostics(scores: pd.DataFrame) -> dict[str, Any]:
    """Why a router generalises or not: fit on train, per-benchmark calibration, within-benchmark
    discrimination. A router whose mean score per training benchmark tracks that benchmark's base
    rate while its held-out AUROC is low has learned provenance rather than difficulty."""
    names = [c.removeprefix("score_") for c in scores.columns if c.startswith("score_")]
    train, test = scores[scores["role"] == "train"], scores[scores["role"] == "test"]
    test_groups = test.groupby("benchmark")
    weights = test_groups.size().to_numpy()
    return {
        "train_in_sample_auroc": {n: auroc(train["label"], train[f"score_{n}"]) for n in names},
        "mean_score_vs_base_rate": {
            str(b): {
                "role": str(g["role"].iloc[0]),
                "base_rate": base_rate(g["label"]),
                **{n: float(g[f"score_{n}"].mean()) for n in names},
            }
            for b, g in scores.groupby("benchmark")
        },
        "test_within_benchmark_auroc_weighted": {
            n: float(
                np.average(
                    [auroc(g["label"], g[f"score_{n}"]) for _, g in test_groups], weights=weights
                )
            )
            for n in names
        },
    }


def _prepare(data: RouterData, attempt: Attempt) -> RouterData:
    """Restrict training to the attempt's benchmarks and attach its training weights."""
    train = data.train
    if attempt.train_benchmarks is not None:
        missing = set(attempt.train_benchmarks) - set(train["benchmark"])
        if missing:
            raise ValueError(f"attempt {attempt.number} needs labels for {sorted(missing)}")
        train = train[train["benchmark"].isin(attempt.train_benchmarks)].reset_index(drop=True)
    if attempt.balanced:
        train = train.assign(weight=balanced_weights(train).to_numpy())
    return RouterData(train=train, val=data.val, test=data.test)


def evaluate(settings: Settings, attempt: Attempt = ATTEMPTS[1]) -> dict[str, Any]:
    seed = settings.seed
    data = _prepare(load_router_data(settings), attempt)
    encoder = FrozenEncoder.from_settings(settings)
    routers: list[Router] = [HeuristicRouter(), RandomRouter(seed), V0Router(encoder, seed=seed)]
    if attempt.with_v1:
        backbone = pretrained_backbone(settings.router_encoder_id, settings.router_encoder_revision)
        routers += [
            V1Router(backbone, seed=s, max_tokens=settings.router_max_tokens) for s in V1_SEEDS
        ]
    every = data.all()
    scores = every[["benchmark", "item_id", "role", "label"]].copy()
    test_y = data.test["label"].to_numpy(dtype=int)
    test_p: dict[str, np.ndarray] = {}

    results: dict[str, Any] = {"routers": {}}
    for router in routers:
        router.fit(data.train, data.val)
        entry: dict[str, Any] = {}
        for split, frame in (("val", data.val), ("test", data.test)):
            p = router.predict_proba(list(frame["router_text"]))
            entry[split] = _split_metrics(frame["label"].to_numpy(dtype=int), p, seed)
            if split == "test":
                test_p[router.name] = p
                entry["test_per_benchmark"] = _per_benchmark(frame, p, seed)
        if isinstance(router, V0Router):
            entry["chosen_c"] = router.chosen_c
            entry["val_auroc_by_c"] = {str(c): a for c, a in router.val_auroc_by_c.items()}
        if isinstance(router, V1Router):
            raw_test = router.predict_proba_uncalibrated(list(data.test["router_text"]))
            entry["v1"] = {
                "seed": router.seed,
                "best_epoch": router.best_epoch,
                "temperature": router.temperature,
                "history": router.history,
                "test_ece_uncalibrated": ece(test_y, raw_test),
                "test_brier_uncalibrated": brier(test_y, raw_test),
            }
        scores[f"score_{router.name}"] = router.predict_proba(list(every["router_text"]))
        results["routers"][router.name] = entry
        log.info(
            "router_evaluated", router=router.name, test_auroc=round(entry["test"]["auroc"], 4)
        )

    comparisons = {}
    pairs = [("v0", "heuristic"), ("v0", "random")]
    v1_names = [r.name for r in routers if isinstance(r, V1Router)]
    pairs += [(n, other) for n in v1_names for other in ("v0", "heuristic")]
    for a, b in pairs:
        diff, lo, hi = paired_auroc_difference_ci(test_y, test_p[a], test_p[b], seed=seed)
        comparisons[f"{a}_minus_{b}"] = {"auroc_diff": diff, "ci95": [lo, hi]}
    results["test_comparisons"] = comparisons

    if v1_names:
        aucs = [results["routers"][n]["test"]["auroc"] for n in v1_names]
        mean, spread = float(np.mean(aucs)), float(max(aucs) - min(aucs))
        gain = mean - results["routers"]["v0"]["test"]["auroc"]
        results["v1"] = {
            "seeds": {n: results["routers"][n].pop("v1") for n in v1_names},
            "test_auroc_mean": mean,
            "test_auroc_std": float(np.std(aucs)),
            "test_auroc_spread": spread,
            "gain_over_v0": gain,
            # Pre-registered: v1 is better than v0 only if the gain exceeds the seed spread.
            "beats_v0": gain > spread,
        }

    v0 = next(r for r in routers if isinstance(r, V0Router))
    assert v0.chosen_c is not None
    results["row_split_optimistic"] = _row_split(
        data, v0.chosen_c, encoder, seed, balanced=attempt.balanced
    )

    lengths = encoder.token_lengths(list(every["router_text"]))
    truncated = lengths > settings.router_max_tokens
    results["truncation"] = {
        "max_tokens": settings.router_max_tokens,
        "overall_rate": float(truncated.mean()),
        "n_truncated": int(truncated.sum()),
        "per_benchmark": {
            str(b): float(truncated[idx].mean())
            for b, idx in every.groupby("benchmark").indices.items()
        },
    }

    results["diagnostics"] = _diagnostics(scores)

    v0_test = results["routers"]["v0"]["test"]
    results["kill_gate"] = {
        "criterion_3_auroc_floor": AUROC_FLOOR,
        "criterion_3_v0_test_auroc_ci_low": v0_test["auroc_ci95"][0],
        "criterion_3_pass": v0_test["auroc_ci95"][0] > AUROC_FLOOR,
        "criteria_1_2": "pending: need frontier answers and the cost-quality sweep",
    }
    if v1_names:
        lows = [results["routers"][n]["test"]["auroc_ci95"][0] for n in v1_names]
        results["kill_gate"]["criterion_3_v1_ci_lows"] = lows
        # Every seed must clear the floor: no picking the luckiest seed.
        results["kill_gate"]["criterion_3_v1_pass_all_seeds"] = all(x > AUROC_FLOOR for x in lows)
    results["attempt"] = {
        "number": attempt.number,
        "train_benchmarks": sorted(data.train["benchmark"].unique()),
        "balanced_weights": attempt.balanced,
        "with_v1": attempt.with_v1,
    }
    results["setup"] = {
        "encoder": settings.router_encoder_id,
        "encoder_revision": settings.router_encoder_revision,
        "max_tokens": settings.router_max_tokens,
        "seed": seed,
        "heuristic_keywords": list(KEYWORDS),
        "split_sizes": {"train": len(data.train), "val": len(data.val), "test": len(data.test)},
        "software": {
            "python": platform.python_version(),
            "switchboard_commit": _git_commit(),
            **{pkg: version(pkg) for pkg in ("torch", "transformers", "scikit-learn", "numpy")},
        },
    }

    stem = attempt.stem
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / f"{stem}.json").write_text(json.dumps(results, indent=2) + "\n")
    scores.to_parquet(settings.results_dir / f"{stem}-scores.parquet", index=False)
    begin, end = f"<!-- BEGIN GENERATED: {stem} -->", f"<!-- END GENERATED: {stem} -->"
    write_into_report(settings.reports_dir / "R2-kill-gate.md", to_markdown(results), begin, end)
    return results


def _ci(v: list[float]) -> str:
    return f"[{v[0]:.3f}, {v[1]:.3f}]"


def to_markdown(r: dict[str, Any]) -> str:
    s = r["setup"]
    lines = [
        f"Encoder `{s['encoder']}` @ `{s['encoder_revision'][:10]}` (frozen for v0) · seed "
        f"{s['seed']} · split sizes train {s['split_sizes']['train']}, val "
        f"{s['split_sizes']['val']}, test {s['split_sizes']['test']}",
        "",
        "**Dataset-level split** (the headline): routers trained on train benchmarks, C chosen on "
        "validation, scored on held-out test benchmarks. 95% bootstrap intervals.",
        "",
        "| Router | Val AUROC | Test AUROC | Test 95% CI | Test ECE | Test Brier |",
        "| --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for name, e in r["routers"].items():
        lines.append(
            f"| {name} | {e['val']['auroc']:.3f} | **{e['test']['auroc']:.3f}** | "
            f"{_ci(e['test']['auroc_ci95'])} | {e['test']['ece']:.3f} | {e['test']['brier']:.3f} |"
        )
    test_base = r["routers"]["v0"]["test"]["base_rate"]
    lines += [
        "",
        f"Test base rate: {100 * test_base:.1f}% (the local model is right this often).",
        "",
    ]

    lines += [
        "**v0 against the baselines on test** (paired bootstrap):",
        "",
        "| Comparison | AUROC difference | 95% CI |",
        "| --- | ---: | --- |",
    ]
    for k, v in r["test_comparisons"].items():
        lines.append(f"| {k.replace('_', ' ')} | {v['auroc_diff']:+.3f} | {_ci(v['ci95'])} |")

    lines += [
        "",
        "**Per test benchmark** (where does prompt-only routing work?):",
        "",
        "| Benchmark | n | Base rate | Heuristic | Random | v0 | v0 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    per = {k: r["routers"][k]["test_per_benchmark"] for k in ("heuristic", "random", "v0")}
    for b, v in per["v0"].items():
        heuristic, rand = per["heuristic"][b]["auroc"], per["random"][b]["auroc"]
        lines.append(
            f"| {b} | {v['n']} | {100 * v['base_rate']:.1f}% | {heuristic:.3f} | {rand:.3f} | "
            f"{v['auroc']:.3f} | {_ci(v['auroc_ci95'])} |"
        )

    rs = r["row_split_optimistic"]
    lines += [
        "",
        "**Row-level split — OPTIMISTIC, for comparison only** (random 80/20 over all rows; the "
        "router can learn which benchmark a prompt came from):",
        "",
        "| Router | Row-split AUROC | Dataset-split AUROC | Gap |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("heuristic", "random", "v0"):
        row, ds = rs[name]["auroc"], r["routers"][name]["test"]["auroc"]
        lines.append(f"| {name} | {row:.3f} | {ds:.3f} | {row - ds:+.3f} |")

    d = r["diagnostics"]
    lines += [
        "",
        "**Diagnostics** — fit on the training benchmarks, and mean score per benchmark "
        "against the local model's actual base rate:",
        "",
        "| Router | Train AUROC (in-sample) | Test AUROC, pooled | Test AUROC, within-benchmark |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in ("heuristic", "random", "v0"):
        lines.append(
            f"| {name} | {d['train_in_sample_auroc'][name]:.3f} | "
            f"{r['routers'][name]['test']['auroc']:.3f} | "
            f"{d['test_within_benchmark_auroc_weighted'][name]:.3f} |"
        )
    lines += [
        "",
        "| Benchmark | Role | Base rate | Heuristic mean score | v0 mean score |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    order = {"train": 0, "val": 1, "test": 2}
    per_bench = d["mean_score_vs_base_rate"].items()
    for b, m in sorted(per_bench, key=lambda kv: order[kv[1]["role"]]):
        lines.append(
            f"| {b} | {m['role']} | {m['base_rate']:.3f} | {m['heuristic']:.3f} | {m['v0']:.3f} |"
        )

    t = r["truncation"]
    lines += [
        "",
        f"Prompts longer than the encoder's {t['max_tokens']}-token limit: "
        f"{t['n_truncated']} ({100 * t['overall_rate']:.2f}%).",
    ]

    g = r["kill_gate"]
    verdict = "PASS" if g["criterion_3_pass"] else "FAIL"
    lines += [
        "",
        f"**Kill gate, criterion 3** (v0 test AUROC 95% CI low > {g['criterion_3_auroc_floor']}): "
        f"low = {g['criterion_3_v0_test_auroc_ci_low']:.3f} → **{verdict}**. Criteria 1 and 2: "
        "pending the frontier answers and the cost-quality sweep.",
    ]
    if "v1" in r:
        lines += _v1_markdown(r)
    return "\n".join(lines)


def _v1_markdown(r: dict[str, Any]) -> list[str]:
    v1 = r["v1"]
    lines = [
        "",
        f"**v1 — fine-tuned encoder, {len(v1['seeds'])} seeds** (benchmark-balanced weighted BCE; "
        "best epoch by validation AUROC; temperature fitted on validation only):",
        "",
        "| Seed | Best epoch | Temperature | Val AUROC | Test AUROC | Test 95% CI | "
        "Test ECE before → after | Test Brier before → after |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for name, s in v1["seeds"].items():
        e = r["routers"][name]
        lines.append(
            f"| {name} | {s['best_epoch']} | {s['temperature']:.2f} | {e['val']['auroc']:.3f} | "
            f"**{e['test']['auroc']:.3f}** | {_ci(e['test']['auroc_ci95'])} | "
            f"{s['test_ece_uncalibrated']:.3f} → {e['test']['ece']:.3f} | "
            f"{s['test_brier_uncalibrated']:.3f} → {e['test']['brier']:.3f} |"
        )
    verdict = "YES" if v1["beats_v0"] else "NO"
    lines += [
        "",
        f"v1 test AUROC mean {v1['test_auroc_mean']:.3f}, spread across seeds "
        f"{v1['test_auroc_spread']:.3f}; gain over v0 {v1['gain_over_v0']:+.3f}. "
        f"Gain larger than the seed spread (pre-registered test of v1 over v0): **{verdict}**.",
        "",
        "| Test benchmark | v0 | " + " | ".join(v1["seeds"]) + " |",
        "| --- | ---: |" + " ---: |" * len(v1["seeds"]),
    ]
    for b, v in r["routers"]["v0"]["test_per_benchmark"].items():
        cells = " | ".join(
            f"{r['routers'][n]['test_per_benchmark'][b]['auroc']:.3f}" for n in v1["seeds"]
        )
        lines.append(f"| {b} | {v['auroc']:.3f} | {cells} |")
    g = r["kill_gate"]
    v1_verdict = "PASS" if g["criterion_3_v1_pass_all_seeds"] else "FAIL"
    lows = ", ".join(f"{x:.3f}" for x in g["criterion_3_v1_ci_lows"])
    lines += [
        "",
        f"**Kill gate, criterion 3 for v1** (every seed's 95% CI low > "
        f"{g['criterion_3_auroc_floor']}): lows = {lows} → **{v1_verdict}**.",
    ]
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the routers for one attempt.")
    parser.add_argument("--attempt", type=int, default=1, choices=sorted(ATTEMPTS))
    args = parser.parse_args(argv)
    configure_logging()
    evaluate(get_settings(), ATTEMPTS[args.attempt])
    return 0


if __name__ == "__main__":
    sys.exit(main())
