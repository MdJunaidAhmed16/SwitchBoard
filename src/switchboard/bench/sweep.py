"""Threshold sweep → cost-quality curves and kill-gate criteria 1-2 (pre-registered in R2).

    python -m switchboard.bench.sweep        (make sweep)

For every router and τ = 0.00 … 1.00, prompts with score ≥ τ go to the local model and the rest
to the frontier. Quality is the fraction answered correctly (local label if routed local, frontier
label if escalated); cost uses ``bench/cost.py``. Everything is pure arithmetic over cached
answers: nothing is regenerated per threshold (08-evaluation, protocol step 5).

Writes ``results/sweep-attempt2.json``, ``reports/figures/cost-quality-attempt2.png`` and the
generated block in ``reports/R2-kill-gate.md``.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from switchboard.bench.cost import frontier_prices, local_prices
from switchboard.config import Settings, get_settings
from switchboard.labeling.generate import labels_path, model_slug
from switchboard.labeling.summary import write_into_report
from switchboard.log import configure_logging, get_logger

log = get_logger(__name__)

TAUS: NDArray[np.float64] = np.round(np.arange(0, 101) / 100, 2)
RETENTION_LEVELS: tuple[float, ...] = tuple(np.round(np.arange(90, 100) / 100, 2))
ESCALATION_LEVELS: tuple[float, ...] = tuple(np.round(np.arange(1, 10) / 10, 1))
QUALITY_FLOOR = 0.95
SENSITIVITY_SCALES: tuple[float, ...] = (0.0, 1.0, 5.0)
BEGIN, END = "<!-- BEGIN GENERATED: sweep-attempt2 -->", "<!-- END GENERATED: sweep-attempt2 -->"


@dataclass(frozen=True)
class EvalSet:
    """Per-prompt facts for the prompts that have both a local and a frontier answer."""

    benchmark: NDArray[np.str_]
    local_ok: NDArray[np.bool_]
    frontier_ok: NDArray[np.bool_]
    local_cost: NDArray[np.float64]
    frontier_cost: NDArray[np.float64]
    scores: dict[str, NDArray[np.float64]]

    def take(self, idx: NDArray[np.int64]) -> EvalSet:
        return EvalSet(
            self.benchmark[idx],
            self.local_ok[idx],
            self.frontier_ok[idx],
            self.local_cost[idx],
            self.frontier_cost[idx],
            {k: v[idx] for k, v in self.scores.items()},
        )


def sweep(ev: EvalSet, scores: NDArray[np.float64]) -> pd.DataFrame:
    """One row per τ: escalation rate, quality and total cost."""
    local = scores[None, :] >= TAUS[:, None]  # shape (taus, prompts)
    correct = np.where(local, ev.local_ok[None, :], ev.frontier_ok[None, :])
    cost = np.where(local, ev.local_cost[None, :], ev.frontier_cost[None, :])
    return pd.DataFrame(
        {
            "tau": TAUS,
            "escalation_rate": 1.0 - local.mean(axis=1),
            "quality": correct.mean(axis=1),
            "cost": cost.sum(axis=1),
        }
    )


def cost_at_retention(curve: pd.DataFrame, frontier_quality: float, level: float) -> float:
    """Lowest cost over τ whose quality retains ``level`` of always-frontier; NaN if none."""
    ok = curve[curve["quality"] >= level * frontier_quality - 1e-12]
    return float(ok["cost"].min()) if len(ok) else float("nan")


def quality_at_escalation(curve: pd.DataFrame, rate: float) -> float:
    """Quality interpolated along the sweep at a given escalation rate."""
    ordered = curve.sort_values("escalation_rate")
    return float(np.interp(rate, ordered["escalation_rate"], ordered["quality"]))


def criterion_1(router: pd.DataFrame, heuristic: pd.DataFrame, fq: float) -> dict[str, Any]:
    rows = []
    for level in RETENTION_LEVELS:
        r, h = cost_at_retention(router, fq, level), cost_at_retention(heuristic, fq, level)
        rows.append({"retention": level, "router_cost": r, "heuristic_cost": h})
    comparable = [x for x in rows if not np.isnan(x["heuristic_cost"])]
    not_worse = all(
        not np.isnan(x["router_cost"]) and x["router_cost"] <= x["heuristic_cost"]
        for x in comparable
    )
    better = any(x["router_cost"] < x["heuristic_cost"] for x in comparable)
    return {"levels": rows, "dominates": bool(comparable) and not_worse and better}


def criterion_2(router: pd.DataFrame, random: pd.DataFrame) -> dict[str, Any]:
    rows = [
        {
            "escalation_rate": e,
            "router_quality": quality_at_escalation(router, e),
            "random_quality": quality_at_escalation(random, e),
        }
        for e in ESCALATION_LEVELS
    ]
    return {
        "levels": rows,
        "beats_random": all(x["router_quality"] > x["random_quality"] for x in rows),
    }


def operating_point(curve: pd.DataFrame, fq: float, always_frontier_cost: float) -> dict[str, Any]:
    ok = curve[curve["quality"] >= QUALITY_FLOOR * fq - 1e-12]
    if not len(ok):
        return {"found": False}
    best = ok.iloc[int(np.argmin(ok["cost"].to_numpy()))]
    return {
        "found": True,
        "tau": float(best["tau"]),
        "share_kept_local": float(1 - best["escalation_rate"]),
        "quality_retention": float(best["quality"] / fq),
        "cost_share_of_always_frontier": float(best["cost"] / always_frontier_cost),
    }


def bootstrap_cost_diff(
    ev: EvalSet, router: str, n_boot: int = 1000, seed: int = 0, level: float = QUALITY_FLOOR
) -> tuple[float, float, float]:
    """Router minus heuristic cost at ``level`` retention, resampling prompts within benchmarks."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(ev.benchmark == b) for b in np.unique(ev.benchmark)]

    def diff(sub: EvalSet) -> float:
        fq = float(sub.frontier_ok.mean())
        r = cost_at_retention(sweep(sub, sub.scores[router]), fq, level)
        h = cost_at_retention(sweep(sub, sub.scores["heuristic"]), fq, level)
        return r - h

    point = diff(ev)
    draws = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        d = diff(ev.take(idx))
        if not np.isnan(d):
            draws.append(d)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return point, float(lo), float(hi)


def load_eval_set(settings: Settings, local_scale: float = 1.0) -> EvalSet:
    scores = pd.read_parquet(settings.results_dir / "router-attempt2-scores.parquet")
    local = pd.read_parquet(labels_path(settings))
    frontier_file = settings.labels_dir / f"{model_slug(settings.frontier_model_id)}.parquet"
    frontier = pd.read_parquet(frontier_file)
    keys = ["benchmark", "item_id"]
    df = (
        scores[scores["role"] == "test"]
        .merge(local[[*keys, "prompt_tokens", "output_tokens"]], on=keys)
        .merge(
            frontier[[*keys, "label", "prompt_tokens", "output_tokens"]],
            on=keys,
            suffixes=("", "_frontier"),
        )
        .sort_values(keys)
        .reset_index(drop=True)
    )
    lp, fp = local_prices(settings, local_scale), frontier_prices(settings)
    score_cols = [c for c in df.columns if c.startswith("score_")]
    return EvalSet(
        benchmark=df["benchmark"].to_numpy(dtype=str),
        local_ok=df["label"].to_numpy(dtype=bool),
        frontier_ok=df["label_frontier"].to_numpy(dtype=bool),
        local_cost=lp.cost(df["prompt_tokens"], df["output_tokens"]),
        frontier_cost=fp.cost(df["prompt_tokens_frontier"], df["output_tokens_frontier"]),
        scores={c.removeprefix("score_"): df[c].to_numpy(dtype=np.float64) for c in score_cols},
    )


def analyse(ev: EvalSet, with_bootstrap: bool = True) -> dict[str, Any]:
    fq = float(ev.frontier_ok.mean())
    always_frontier = float(ev.frontier_cost.sum())
    curves = {name: sweep(ev, s) for name, s in ev.scores.items()}
    learned = [n for n in curves if n not in ("heuristic", "random")]
    out: dict[str, Any] = {
        "n_prompts": len(ev.local_ok),
        "n_by_benchmark": {str(b): int((ev.benchmark == b).sum()) for b in np.unique(ev.benchmark)},
        "always_frontier": {"quality": fq, "cost": always_frontier},
        "always_local": {
            "quality": float(ev.local_ok.mean()),
            "cost": float(ev.local_cost.sum()),
            "quality_retention": float(ev.local_ok.mean() / fq),
        },
        "routers": {},
    }
    for name, curve in curves.items():
        entry: dict[str, Any] = {
            "operating_point": operating_point(curve, fq, always_frontier),
            "curve": curve.to_dict(orient="list"),
        }
        if name in learned:
            entry["criterion_1"] = criterion_1(curve, curves["heuristic"], fq)
            entry["criterion_2"] = criterion_2(curve, curves["random"])
            if with_bootstrap:
                d, lo, hi = bootstrap_cost_diff(ev, name)
                entry["cost_vs_heuristic_at_95"] = {"diff_usd": d, "ci95": [lo, hi]}
        out["routers"][name] = entry
    return out


def _fmt_usd_per_1k(cost: float, n: int) -> str:
    return f"${1000 * cost / n:.2f}"


def to_markdown(main: dict[str, Any], sensitivity: Mapping[str, dict[str, Any]]) -> str:
    n = main["n_prompts"]
    af, al = main["always_frontier"], main["always_local"]
    lines = [
        f"Evaluation subset: {n} test prompts "
        f"({', '.join(f'{b} {k}' for b, k in main['n_by_benchmark'].items())}).",
        "",
        f"- **Always-frontier:** quality {100 * af['quality']:.1f}%, cost "
        f"{_fmt_usd_per_1k(af['cost'], n)} per 1,000 prompts.",
        f"- **Always-local:** quality {100 * al['quality']:.1f}% "
        f"({100 * al['quality_retention']:.1f}% of always-frontier), cost "
        f"{_fmt_usd_per_1k(al['cost'], n)} per 1,000 prompts.",
        "",
        "**Operating point** — cheapest τ retaining ≥ 95% of always-frontier quality (chosen and "
        "reported on this subset: in-sample):",
        "",
        "| Router | τ | Kept local | Quality retained | Cost vs always-frontier |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, e in main["routers"].items():
        op = e["operating_point"]
        if op["found"]:
            lines.append(
                f"| {name} | {op['tau']:.2f} | {100 * op['share_kept_local']:.1f}% | "
                f"{100 * op['quality_retention']:.1f}% | "
                f"{100 * op['cost_share_of_always_frontier']:.1f}% |"
            )
        else:
            lines.append(f"| {name} | — | — | never reaches 95% | — |")
    lines += [
        "",
        "**Gate criteria 1-2** (pre-registered definitions):",
        "",
        "| Router | C1: dominates heuristic, 90-99% retention | C2: beats random at matched "
        "escalation | Cost - heuristic at 95% (per 1,000 prompts, 95% CI) |",
        "| --- | --- | --- | --- |",
    ]
    for name, e in main["routers"].items():
        if "criterion_1" not in e:
            continue
        c = e.get("cost_vs_heuristic_at_95")
        diff = (
            (
                f"{1000 * c['diff_usd'] / n:+.2f} [{1000 * c['ci95'][0] / n:+.2f}, "
                f"{1000 * c['ci95'][1] / n:+.2f}]"
            )
            if c
            else "—"
        )
        lines.append(
            f"| {name} | {'PASS' if e['criterion_1']['dominates'] else 'FAIL'} | "
            f"{'PASS' if e['criterion_2']['beats_random'] else 'FAIL'} | {diff} |"
        )
    lines += [
        "",
        "**Sensitivity to the local price** (criterion 1 / criterion 2 per router):",
        "",
        "| Local price x | "
        + " | ".join(k for k in main["routers"] if "criterion_1" in main["routers"][k])
        + " |",
        "| --- |" + " --- |" * sum("criterion_1" in e for e in main["routers"].values()),
    ]
    for scale, res in sensitivity.items():
        cells = [
            f"{'P' if e['criterion_1']['dominates'] else 'F'}/"
            f"{'P' if e['criterion_2']['beats_random'] else 'F'}"
            for e in res["routers"].values()
            if "criterion_1" in e
        ]
        lines.append(f"| {scale} | " + " | ".join(cells) + " |")
    lines += ["", "Chart: `reports/figures/cost-quality-attempt2.png`."]
    return "\n".join(lines)


def plot(main: dict[str, Any], path: Any) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n, fq = main["n_prompts"], main["always_frontier"]["quality"]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    styles = {
        "heuristic": ("tab:orange", "-", 2.0),
        "random": ("tab:gray", ":", 1.5),
        "v0": ("tab:blue", "--", 1.5),
    }
    for name, e in main["routers"].items():
        c = e["curve"]
        color, ls, lw = styles.get(name, ("tab:green", "-", 1.2))
        ax.plot(
            1000 * np.array(c["cost"]) / n,
            100 * np.array(c["quality"]) / fq,
            ls,
            color=color,
            lw=lw,
            label=name,
            alpha=0.9 if name in styles else 0.7,
        )
    af, al = main["always_frontier"], main["always_local"]
    ax.scatter([1000 * af["cost"] / n], [100.0], color="black", zorder=5, label="always-frontier")
    ax.scatter(
        [1000 * al["cost"] / n],
        [100 * al["quality_retention"]],
        color="black",
        marker="x",
        zorder=5,
        label="always-local",
    )
    ax.axhline(95, color="lightgray", lw=1)
    ax.set_xlabel("Cost per 1,000 prompts (USD)")
    ax.set_ylabel("Quality retained (% of always-frontier)")
    ax.set_title(f"Cost vs quality — attempt 2, {n}-prompt test subset")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    configure_logging()
    settings = get_settings()
    main_result = analyse(load_eval_set(settings))
    sensitivity = {
        str(s): analyse(load_eval_set(settings, s), with_bootstrap=False)
        for s in SENSITIVITY_SCALES
    }
    out = {
        "main": main_result,
        "sensitivity": {
            k: {
                n: {
                    "criterion_1": e["criterion_1"]["dominates"],
                    "criterion_2": e["criterion_2"]["beats_random"],
                }
                for n, e in v["routers"].items()
                if "criterion_1" in e
            }
            for k, v in sensitivity.items()
        },
        "prices": {
            "local_per_mtok": [settings.local_price_in_per_mtok, settings.local_price_out_per_mtok],
            "frontier_per_mtok": [
                settings.frontier_price_in_per_mtok,
                settings.frontier_price_out_per_mtok,
            ],
        },
    }
    (settings.results_dir / "sweep-attempt2.json").write_text(json.dumps(out, indent=2) + "\n")
    plot(main_result, settings.reports_dir / "figures" / "cost-quality-attempt2.png")
    write_into_report(
        settings.reports_dir / "R2-kill-gate.md", to_markdown(main_result, sensitivity), BEGIN, END
    )
    log.info("sweep_done", n_prompts=main_result["n_prompts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
