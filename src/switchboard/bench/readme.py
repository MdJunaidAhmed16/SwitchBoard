"""README results (12-demo-and-readme). Every number in ``README.md`` between GENERATED markers is
written here from the committed results files; none is typed by hand.

    python -m switchboard.bench.readme        (make readme)

Reads ``results/router-v0.json`` (attempt 1), ``results/router-attempt2.json``,
``results/router-attempt2-scores.parquet``, ``results/sweep-attempt2.json`` and
``results/labels-summary.json``. Writes the ``readme-headline`` and ``readme-results`` blocks and
``reports/figures/reliability-attempt2.png``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from switchboard.config import REPO_ROOT, get_settings
from switchboard.labeling.summary import write_into_report
from switchboard.log import configure_logging, get_logger

log = get_logger(__name__)

V1_SEEDS = ("v1_s0", "v1_s1", "v1_s2")
LEARNED = ("v0", *V1_SEEDS)
RELIABILITY_BINS = 10
EN_DASH = chr(0x2013)  # an en dash, spelled out for ruff RUF001
MIN_BIN_COUNT = 20  # sparser bins are noise, not calibration; dropped from the diagram
# Categorical slots 1-3 of the dataviz reference palette (validated, light surface).
COLORS = {"heuristic": "#2a78d6", "v0": "#eb6834", "v1": "#1baf7a"}


def passes(entry: dict[str, Any]) -> tuple[bool, bool]:
    """Gate criteria 1 (dominates the heuristic) and 2 (beats random) for one router."""
    return bool(entry["criterion_1"]["dominates"]), bool(entry["criterion_2"]["beats_random"])


def _block(name: str) -> tuple[str, str]:
    return f"<!-- BEGIN GENERATED: {name} -->", f"<!-- END GENERATED: {name} -->"


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}%"


def _span(values: list[float], fmt: Any = _pct) -> str:
    lo, hi = min(values), max(values)
    return fmt(lo) if fmt(lo) == fmt(hi) else f"{fmt(lo)}{EN_DASH}{fmt(hi)}"


def _ci(stats: dict[str, Any], key: str = "auroc") -> str:
    lo, hi = stats[f"{key}_ci95"]
    return f"{stats[key]:.3f} [{lo:.3f}, {hi:.3f}]"


def _per_1k(cost: float, n: int) -> str:
    return f"${1000 * cost / n:.2f}"


def headline(sweep: dict[str, Any]) -> str:
    """The README's headline: the 95%-retention operating point of every router, as ranges."""
    routers = sweep["routers"]
    ops = {name: routers[name]["operating_point"] for name in routers}
    v1 = [ops[s] for s in V1_SEEDS]
    diffs = [routers[s]["cost_vs_heuristic_at_95"]["ci95"] for s in LEARNED]
    all_include_zero = all(lo <= 0 <= hi for lo, hi in diffs)
    gate_failed = not any(all(passes(routers[s])) for s in LEARNED)
    return "\n".join(
        [
            "> At 95% of always-frontier quality, the learned router (v1, three seeds) keeps "
            f"**{_span([o['share_kept_local'] for o in v1])}** of traffic on the local model, at "
            f"**{_span([o['cost_share_of_always_frontier'] for o in v1])}** of always-frontier "
            "cost. The length-and-keyword heuristic keeps "
            f"{_pct(ops['heuristic']['share_kept_local'])} at "
            f"{_pct(ops['heuristic']['cost_share_of_always_frontier'])}; random routing "
            f"{_pct(ops['random']['share_kept_local'])} at "
            f"{_pct(ops['random']['cost_share_of_always_frontier'])}.",
            ">",
            "> "
            + (
                "No learned router's cost differs significantly from the heuristic's "
                "(every 95% bootstrap interval includes zero). "
                if all_include_zero
                else ""
            )
            + (
                "**The pre-registered kill gate failed**: no router beats both the heuristic "
                "and random routing on any seed."
                if gate_failed
                else "The pre-registered kill gate passed."
            ),
        ]
    )


def results(
    attempt1: dict[str, Any],
    attempt2: dict[str, Any],
    sweep: dict[str, Any],
    labels: dict[str, Any],
) -> str:
    n = sweep["n_prompts"]
    af, al = sweep["always_frontier"], sweep["always_local"]
    roles = labels["by_role"]
    r2, routers = attempt2["routers"], sweep["routers"]
    row = attempt2["row_split_optimistic"]
    seeds = attempt2["v1"]["seeds"]

    lines = [
        "**Base rates** — how often each model is already right.",
        "",
        "| | Local model (Qwen2.5-1.5B) | Frontier (Claude Opus 5.5) |",
        "| --- | --- | --- |",
        f"| Train benchmarks ({roles['train']['n']:,} prompts) | "
        f"{_pct(roles['train']['base_rate'])} | — |",
        f"| Test benchmarks ({roles['test']['n']:,} prompts) | "
        f"{_pct(roles['test']['base_rate'])} | — |",
        f"| Test subset with frontier answers ({n} prompts) | {_pct(al['quality'])} | "
        f"{_pct(af['quality'])} |",
        f"| Cost per 1,000 prompts on the subset | {_per_1k(al['cost'], n)} | "
        f"{_per_1k(af['cost'], n)} |",
        "",
        f"**Routers** — AUROC on all {roles['test']['n']:,} test prompts (95% bootstrap interval); "
        f"cost and gate criteria on the {n}-prompt subset at 95% quality retention.",
        "",
        "| Router | Test AUROC | ECE | Kept local | Cost vs always-frontier | "
        "C1: beats heuristic | C2: beats random |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("heuristic", "random", *LEARNED):
        op = routers[name]["operating_point"]
        ece = f"{r2[name]['test']['ece']:.3f}"
        if name in seeds:
            ece = f"{seeds[name]['test_ece_uncalibrated']:.3f} → {ece}"
        gate = (
            ["pass" if ok else "**fail**" for ok in passes(routers[name])]
            if name in LEARNED
            else ["—", "—"]
        )
        lines.append(
            f"| {name} | {_ci(r2[name]['test'])} | {ece} | {_pct(op['share_kept_local'])} | "
            f"{_pct(op['cost_share_of_always_frontier'])} | " + " | ".join(gate) + " |"
        )
    v1 = attempt2["v1"]
    a1 = attempt1["routers"]["v0"]["test"]
    lines += [
        "",
        f"v1 mean test AUROC {v1['test_auroc_mean']:.3f} (seed spread "
        f"{v1['test_auroc_spread']:.3f}); ECE for v1 is before → after temperature scaling.",
        "",
        "**Why the split matters** — the same frozen-encoder router (v0), scored two ways.",
        "",
        "| | Dataset-level split (test benchmarks never seen) | Row split — optimistic |",
        "| --- | --- | --- |",
        f"| Attempt 1 (train: GSM8K, MMLU, MBPP) | {_ci(a1)} | "
        f"{_ci(attempt1['row_split_optimistic']['v0'])} |",
        f"| Attempt 2 (nine train benchmarks, balanced weights) | {_ci(r2['v0']['test'])} | "
        f"{_ci(row['v0'])} |",
        "",
        "Charts: `reports/figures/cost-quality-attempt2.png`, "
        "`reports/figures/reliability-attempt2.png`.",
    ]
    return "\n".join(lines)


def reliability_curve(
    y: np.ndarray, p: np.ndarray, bins: int = RELIABILITY_BINS, min_count: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean predicted score, observed accuracy and count for each equal-width bin holding at least
    ``min_count`` prompts (the same bins as ``metrics.ece``)."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    keep = [b for b in range(bins) if (which == b).sum() >= max(min_count, 1)]
    mean_p = np.array([p[which == b].mean() for b in keep])
    acc = np.array([y[which == b].mean() for b in keep])
    count = np.array([(which == b).sum() for b in keep])
    return mean_p, acc, count


def plot_reliability(scores: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    test = scores[scores["role"] == "test"]
    y = test["label"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="#9a9a94", lw=1, ls="--", label="perfectly calibrated")
    series = [("heuristic", "heuristic", "o", "-"), ("v0", "v0", "s", "-")] + [
        (s, f"v1 seed {s[-1]}", m, ls)
        for s, m, ls in zip(V1_SEEDS, "^Dv", ("-", "--", ":"), strict=True)
    ]
    for name, label, marker, ls in series:
        p = test[f"score_{name}"].to_numpy(dtype=float)
        mean_p, acc, _ = reliability_curve(y, p, min_count=MIN_BIN_COUNT)
        color = COLORS["v1" if name.startswith("v1") else name]
        ax.plot(mean_p, acc, ls, marker=marker, ms=6, lw=2, color=color, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability the local model is right")
    ax.set_ylabel("Observed fraction right")
    ax.set_title(
        f"Reliability on the test benchmarks ({len(test):,} prompts)\n"
        f"10 equal-width bins; bins under {MIN_BIN_COUNT} prompts omitted",
        fontsize=10,
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper left", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    configure_logging()
    settings = get_settings()
    res = settings.results_dir

    def load(name: str) -> dict[str, Any]:
        data: dict[str, Any] = json.loads((res / name).read_text())
        return data

    sweep = load("sweep-attempt2.json")["main"]
    readme = REPO_ROOT / "README.md"
    write_into_report(readme, headline(sweep), *_block("readme-headline"))
    block = results(
        load("router-v0.json"), load("router-attempt2.json"), sweep, load("labels-summary.json")
    )
    write_into_report(readme, block, *_block("readme-results"))
    figure = settings.reports_dir / "figures" / "reliability-attempt2.png"
    plot_reliability(pd.read_parquet(res / "router-attempt2-scores.parquet"), figure)
    log.info("readme_done", readme=str(readme), figure=str(figure))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
