"""README results (12-demo-and-readme). Every number in ``README.md`` between GENERATED markers is
written here from the committed results files; none is typed by hand.

    python -m switchboard.bench.readme        (make readme)

Reads, for attempt 1, ``results/router-v0.json`` and, for each attempt in ``SHOWN``,
``results/router-attempt<N>.json``, ``results/router-attempt<N>-scores.parquet`` and
``results/sweep-attempt<N>.json``. Writes the ``readme-headline`` and ``readme-results`` blocks and
``reports/figures/reliability-attempt<N>.png``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from switchboard.config import REPO_ROOT, Settings, get_settings
from switchboard.labeling.summary import write_into_report
from switchboard.log import configure_logging, get_logger
from switchboard.router.attempts import ATTEMPTS

log = get_logger(__name__)

SHOWN = (2, 3)  # attempts with frontier answers, a v1 router and a cost-quality sweep
V1_SEEDS = ("v1_s0", "v1_s1", "v1_s2")
LEARNED = ("v0", *V1_SEEDS)
RELIABILITY_BINS = 10
EN_DASH = chr(0x2013)  # an en dash, spelled out for ruff RUF001
MIN_BIN_COUNT = 20  # sparser bins are noise, not calibration; dropped from the diagram
# Categorical slots 1-3 of the dataviz reference palette (validated, light surface).
COLORS = {"heuristic": "#2a78d6", "v0": "#eb6834", "v1": "#1baf7a"}


@dataclass(frozen=True)
class AttemptResult:
    number: int
    local_model: str
    router: dict[str, Any]  # results/router-attempt<N>.json
    sweep: dict[str, Any]  # the "main" part of results/sweep-attempt<N>.json


def passes(entry: dict[str, Any]) -> tuple[bool, bool]:
    """Gate criteria 1 (dominates the heuristic) and 2 (beats random) for one router."""
    return bool(entry["criterion_1"]["dominates"]), bool(entry["criterion_2"]["beats_random"])


def gate_passed(a: AttemptResult) -> bool:
    """The pre-registered kill gate: a learned router passes criteria 1-3 — v0, or v1 on every
    seed."""
    kg, routers = a.router["kill_gate"], a.sweep["routers"]
    v0 = bool(kg["criterion_3_pass"]) and all(passes(routers["v0"]))
    v1 = bool(kg.get("criterion_3_v1_pass_all_seeds", False)) and all(
        all(passes(routers[s])) for s in V1_SEEDS
    )
    return v0 or v1


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


def _short(model_id: str) -> str:
    return model_id.split("/")[-1]


def _operating(a: AttemptResult, names: tuple[str, ...]) -> str:
    ops = [a.sweep["routers"][n]["operating_point"] for n in names]
    if not all(o["found"] for o in ops):
        return "never reaches 95%"
    kept = _span([o["share_kept_local"] for o in ops])
    cost = _span([o["cost_share_of_always_frontier"] for o in ops])
    return f"{kept} kept local, {cost} of the cost"


def headline(attempts: list[AttemptResult]) -> str:
    """Each attempt's 95%-retention operating points and gate verdict, side by side. v1 is a range
    over its three seeds, never the best seed."""
    cols = [f"Attempt {a.number}: `{_short(a.local_model)}` local" for a in attempts]
    lines = [
        "| At 95% of always-frontier quality | " + " | ".join(cols) + " |",
        "| --- |" + " --- |" * len(attempts),
    ]

    def row(label: str, cells: list[str]) -> None:
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row(
        "Local model right (test subset)",
        [_pct(a.sweep["always_local"]["quality"]) for a in attempts],
    )
    row("Learned router, v1 (3 seeds)", [f"**{_operating(a, V1_SEEDS)}**" for a in attempts])
    row("Length-and-keyword heuristic", [_operating(a, ("heuristic",)) for a in attempts])
    row("Random routing", [_operating(a, ("random",)) for a in attempts])

    def significant(a: AttemptResult) -> str:
        cis = [a.sweep["routers"][s]["cost_vs_heuristic_at_95"]["ci95"] for s in V1_SEEDS]
        cheaper = sum(hi < 0 for _, hi in cis)
        return f"{cheaper} of 3 seeds" if cheaper else "no seed"

    row("v1 significantly cheaper than the heuristic", [significant(a) for a in attempts])
    row(
        "Pre-registered kill gate",
        ["passed" if gate_passed(a) else "**failed**" for a in attempts],
    )
    return "\n".join(lines)


def router_table(a: AttemptResult) -> list[str]:
    r, routers = a.router["routers"], a.sweep["routers"]
    seeds = a.router["v1"]["seeds"]
    n = a.sweep["n_prompts"]
    af, al = a.sweep["always_frontier"], a.sweep["always_local"]
    lines = [
        f"**Attempt {a.number} — local model `{_short(a.local_model)}`.** Right on "
        f"{_pct(r['heuristic']['test']['base_rate'])} of all {r['heuristic']['test']['n']:,} test "
        f"prompts; on the {n}-prompt subset {_pct(al['quality'])} at "
        f"{_per_1k(al['cost'], n)} per 1,000 prompts, against the frontier's {_pct(af['quality'])} "
        f"at {_per_1k(af['cost'], n)}.",
        "",
        "| Router | Test AUROC | ECE | Kept local | Cost vs always-frontier | "
        "C1: beats heuristic | C2: beats random |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in ("heuristic", "random", *LEARNED):
        op = routers[name]["operating_point"]
        ece = f"{r[name]['test']['ece']:.3f}"
        if name in seeds:
            ece = f"{seeds[name]['test_ece_uncalibrated']:.3f} → {ece}"
        gate = (
            ["pass" if ok else "**fail**" for ok in passes(routers[name])]
            if name in LEARNED
            else ["—", "—"]
        )
        kept = _pct(op["share_kept_local"]) if op["found"] else "—"
        cost = _pct(op["cost_share_of_always_frontier"]) if op["found"] else "never 95%"
        lines.append(
            f"| {name} | {_ci(r[name]['test'])} | {ece} | {kept} | {cost} | "
            + " | ".join(gate)
            + " |"
        )
    v1 = a.router["v1"]
    lines += [
        "",
        f"v1 mean test AUROC {v1['test_auroc_mean']:.3f} (seed spread "
        f"{v1['test_auroc_spread']:.3f}); ECE for v1 is before → after temperature scaling.",
    ]
    return lines


def results(attempt1: dict[str, Any], attempts: list[AttemptResult]) -> str:
    lines = [
        f"AUROC on all test prompts (95% bootstrap interval); cost and gate criteria on the "
        f"{attempts[0].sweep['n_prompts']}-prompt subset with frontier answers, at 95% quality "
        "retention.",
        "",
    ]
    for a in attempts:
        lines += [*router_table(a), ""]
    lines += [
        "**Why the split matters** — the frozen-encoder router (v0), scored two ways.",
        "",
        "| | Dataset-level split (test benchmarks never seen) | Row split — optimistic |",
        "| --- | --- | --- |",
        f"| Attempt 1 (train: GSM8K, MMLU, MBPP) | {_ci(attempt1['routers']['v0']['test'])} | "
        f"{_ci(attempt1['row_split_optimistic']['v0'])} |",
    ]
    for a in attempts:
        lines.append(
            f"| Attempt {a.number} (nine train benchmarks, balanced weights) | "
            f"{_ci(a.router['routers']['v0']['test'])} | "
            f"{_ci(a.router['row_split_optimistic']['v0'])} |"
        )
    charts = ", ".join(
        f"`reports/figures/{kind}-attempt{a.number}.png`"
        for a in attempts
        for kind in ("cost-quality", "reliability")
    )
    lines += ["", f"Charts: {charts}."]
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


def plot_reliability(scores: pd.DataFrame, path: Path, title: str) -> None:
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
        f"{title}: reliability on the test benchmarks ({len(test):,} prompts)\n"
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


def load_attempt(settings: Settings, number: int) -> AttemptResult:
    res = settings.results_dir
    return AttemptResult(
        number=number,
        local_model=ATTEMPTS[number].settings_for(settings).local_model_id,
        router=json.loads((res / f"router-attempt{number}.json").read_text()),
        sweep=json.loads((res / f"sweep-attempt{number}.json").read_text())["main"],
    )


def main() -> int:
    configure_logging()
    settings = get_settings()
    attempts = [load_attempt(settings, n) for n in SHOWN]
    attempt1 = json.loads((settings.results_dir / "router-v0.json").read_text())
    readme = REPO_ROOT / "README.md"
    write_into_report(readme, headline(attempts), *_block("readme-headline"))
    write_into_report(readme, results(attempt1, attempts), *_block("readme-results"))
    for a in attempts:
        figure = settings.reports_dir / "figures" / f"reliability-attempt{a.number}.png"
        scores = pd.read_parquet(settings.results_dir / f"router-attempt{a.number}-scores.parquet")
        plot_reliability(scores, figure, f"Attempt {a.number}, {_short(a.local_model)}")
    log.info("readme_done", readme=str(readme), attempts=list(SHOWN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
