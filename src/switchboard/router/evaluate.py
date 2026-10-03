"""Phase 2 router evaluation: heuristic, random and v0 on the dataset-level split.

    python -m switchboard.router.evaluate        (make train-v0)

Writes ``results/router-v0.json`` (every number), ``results/router-v0-scores.parquet`` (one score
per router per labelled prompt, for the cost-quality sweep) and the generated tables in
``reports/R2-kill-gate.md``.

What this run can and cannot decide. Kill-gate criterion 3 (AUROC meaningfully above 0.5) is
evaluated here. Criteria 1 and 2 (v0's cost-quality curve dominates the heuristic's and beats random
routing at matched escalation) need the frontier model's answers and the cost model, and are
evaluated by the sweep. The AUROC comparisons below are diagnostics for those, not substitutes.
"""

from __future__ import annotations

import json
import platform
import sys
from importlib.metadata import version
from typing import Any

import numpy as np
import pandas as pd

from switchboard.config import Settings, get_settings
from switchboard.labeling.generate import _git_commit
from switchboard.labeling.summary import write_into_report
from switchboard.log import configure_logging, get_logger
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

log = get_logger(__name__)

# Kill-gate criterion 3, fixed before the first evaluation run: the lower end of v0's 95%
# bootstrap interval for test AUROC must clear this, not merely 0.5.
AUROC_FLOOR = 0.55

BEGIN = "<!-- BEGIN GENERATED: router-v0 -->"
END = "<!-- END GENERATED: router-v0 -->"


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


def _row_split(data: RouterData, v0_c: float, encoder: FrozenEncoder, seed: int) -> dict[str, Any]:
    """Optimistic comparison: random 80/20 split over all rows, same model settings."""
    train, test = row_split(data.all(), seed=seed)
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


def evaluate(settings: Settings) -> dict[str, Any]:
    seed = settings.seed
    data = load_router_data(settings)
    encoder = FrozenEncoder.from_settings(settings)
    routers: list[Router] = [HeuristicRouter(), RandomRouter(seed), V0Router(encoder, seed=seed)]
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
        scores[f"score_{router.name}"] = router.predict_proba(list(every["router_text"]))
        results["routers"][router.name] = entry
        log.info(
            "router_evaluated", router=router.name, test_auroc=round(entry["test"]["auroc"], 4)
        )

    comparisons = {}
    for other in ("heuristic", "random"):
        diff, lo, hi = paired_auroc_difference_ci(test_y, test_p["v0"], test_p[other], seed=seed)
        comparisons[f"v0_minus_{other}"] = {"auroc_diff": diff, "ci95": [lo, hi]}
    results["test_comparisons"] = comparisons

    v0 = next(r for r in routers if isinstance(r, V0Router))
    assert v0.chosen_c is not None
    results["row_split_optimistic"] = _row_split(data, v0.chosen_c, encoder, seed)

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

    v0_test = results["routers"]["v0"]["test"]
    results["kill_gate"] = {
        "criterion_3_auroc_floor": AUROC_FLOOR,
        "criterion_3_v0_test_auroc_ci_low": v0_test["auroc_ci95"][0],
        "criterion_3_pass": v0_test["auroc_ci95"][0] > AUROC_FLOOR,
        "criteria_1_2": "pending: need frontier answers and the cost-quality sweep",
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

    settings.results_dir.mkdir(parents=True, exist_ok=True)
    (settings.results_dir / "router-v0.json").write_text(json.dumps(results, indent=2) + "\n")
    scores.to_parquet(settings.results_dir / "router-v0-scores.parquet", index=False)
    write_into_report(settings.reports_dir / "R2-kill-gate.md", to_markdown(results), BEGIN, END)
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
    return "\n".join(lines)


def main() -> int:
    configure_logging()
    evaluate(get_settings())
    return 0


if __name__ == "__main__":
    sys.exit(main())
