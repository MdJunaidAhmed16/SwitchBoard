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
from switchboard.labeling.benchmarks import REGISTRY
from switchboard.labeling.generate import labels_path, model_slug
from switchboard.labeling.graders import CHOICE_BENCHMARKS, CODE_BENCHMARKS, NUMERIC_BENCHMARKS
from switchboard.labeling.splits import load_splits
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


def _grader(name: str) -> str:
    if name in NUMERIC_BENCHMARKS:
        return "exact number"
    if name in CHOICE_BENCHMARKS:
        return "multiple choice"
    if name in CODE_BENCHMARKS:
        return "unit tests (Docker)"
    return {"math": "equivalent final answer", "bbh": "exact answer"}.get(name, "answer match")


def model_accuracy(settings: Settings) -> tuple[dict[str, str], pd.DataFrame]:
    """Per-benchmark accuracy of each local model and the frontier model, from the label files.

    Returns display names and a frame indexed by benchmark with ``n``, one accuracy column per
    model and ``n_frontier`` (the frontier answered only the test subset).
    """
    models = {
        "local_1": ATTEMPTS[2].settings_for(settings),
        "local_2": ATTEMPTS[3].settings_for(settings),
    }
    names = {k: _short(s.local_model_id) for k, s in models.items()}
    frames = {k: pd.read_parquet(labels_path(s)) for k, s in models.items()}
    frontier = pd.read_parquet(
        settings.labels_dir / f"{model_slug(settings.frontier_model_id)}.parquet"
    )
    names["frontier"] = _short(settings.frontier_model_id)
    out = pd.DataFrame({"n": frames["local_1"].groupby("benchmark").size()})
    for key, df in frames.items():
        out[key] = df.groupby("benchmark")["label"].mean()
    out["frontier"] = frontier.groupby("benchmark")["label"].mean()
    out["n_frontier"] = frontier.groupby("benchmark").size()
    return names, out


def benchmark_table(settings: Settings, names: dict[str, str], acc: pd.DataFrame) -> str:
    roles = load_splits(settings.splits_path, known=set(REGISTRY))
    order = {"train": 0, "val": 1, "test": 2}
    rows = sorted(acc.index, key=lambda b: (order[roles[b]], b))
    lines = [
        f"| Benchmark | Domain | Role | Prompts | Graded by | `{names['local_1']}` | "
        f"`{names['local_2']}` | `{names['frontier']}` (test subset) |",
        "| --- | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for b in rows:
        r = acc.loc[b]
        frontier = (
            f"{_pct(r['frontier'])} (n={int(r['n_frontier'])})" if r["n_frontier"] > 0 else "—"
        )
        lines.append(
            f"| [{b}](https://huggingface.co/datasets/{REGISTRY[b].repo_id}) | {REGISTRY[b].track} "
            f"| {roles[b]} | {int(r['n']):,} | {_grader(b)} | {_pct(r['local_1'])} | "
            f"{_pct(r['local_2'])} | {frontier} |"
        )
    total = int(acc["n"].sum())
    lines += [
        "",
        f"{len(rows)} benchmarks, {total:,} prompts per local model. Accuracy is the share of "
        "prompts the model answered correctly — for the router, the base rate it has to beat.",
    ]
    return "\n".join(lines)


def plot_benchmarks(
    settings: Settings, names: dict[str, str], acc: pd.DataFrame, path: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    roles = load_splits(settings.splits_path, known=set(REGISTRY))
    order = {"train": 0, "val": 1, "test": 2}
    rows = sorted(acc.index, key=lambda b: (order[roles[b]], b))
    y = np.arange(len(rows))
    height = 0.27
    series = [("local_1", "#2a78d6"), ("local_2", "#eb6834"), ("frontier", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, (key, color) in enumerate(series):
        values = acc.loc[rows, key].to_numpy(dtype=float)
        ax.barh(y + (i - 1) * height, 100 * np.nan_to_num(values), height * 0.9, color=color,
                label=names[key] + (" (test subset)" if key == "frontier" else ""))  # fmt: skip
    ax.set_yticks(y, [f"{b} ({roles[b]})" for b in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Answered correctly (%)")
    ax.set_title(
        "Accuracy per benchmark: two local models and the frontier model", fontsize=11, pad=30
    )
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def setup_table(settings: Settings) -> str:
    """Every model and encoder used, with its pinned revision and how it ran."""
    lines = [
        "| Component | Model | Revision | How it ran |",
        "| --- | --- | --- | --- |",
    ]
    for number, label in ((2, f"Local model, attempts 1{EN_DASH}2"), (3, "Local model, attempt 3")):
        s = ATTEMPTS[number].settings_for(settings)
        meta = json.loads(labels_path(s).with_suffix(".meta.json").read_text())
        v, d = meta["vllm"], meta["decode_params"]
        lines.append(
            f"| {label} | `{meta['model_id']}` | `{meta['model_revision'][:10]}` | vLLM "
            f"{v['version']} on an RTX 4060 laptop GPU (8 GB), {v['max_num_seqs']} in flight; "
            f"greedy (temperature {d['temperature']}), up to {d['max_tokens']:,} new tokens |"
        )
    lines += [
        f"| Frontier model | `{settings.frontier_model_id}` | via OpenRouter | answers cached "
        f"once; ${settings.frontier_price_in_per_mtok:.2f} / "
        f"${settings.frontier_price_out_per_mtok:.2f} "
        "per million input / output tokens |",
        f"| Router encoder (v0, v1) | `{settings.router_encoder_id}` | "
        f"`{settings.router_encoder_revision[:10]}` | v0: frozen + logistic regression; v1: "
        f"fine-tuned end to end, 3 seeds, temperature-scaled; prompts cut at "
        f"{settings.router_max_tokens} tokens |",
        "| Heuristic baseline | logistic regression | — | log word count + count of 12 "
        "difficulty keywords |",
        "| Random baseline | hash of the prompt | — | uniform score, for matched-rate comparison |",
    ]
    return "\n".join(lines)


def classification(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy, precision, recall and F1 of "the local model will be right" at ``threshold``."""
    pred = p >= threshold
    tp = float((pred & y).sum())
    precision = tp / pred.sum() if pred.any() else float("nan")
    recall = tp / y.sum() if y.any() else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "accuracy": float((pred == y).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_right": float(pred.mean()),
    }


def classification_table(attempts: list[AttemptResult], settings: Settings) -> str:
    lines = [
        "Each router as a yes/no classifier of *the local model will answer this test prompt "
        "correctly*, at a score threshold of 0.5 (calibrated scores), on all test prompts. "
        "Precision is how often a prompt the router keeps local is answered correctly.",
        "",
        "| Attempt (local model) | Router | Accuracy | Precision | Recall | F1 | Predicted right |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for a in attempts:
        scores = pd.read_parquet(settings.results_dir / f"router-attempt{a.number}-scores.parquet")
        test = scores[scores["role"] == "test"]
        y = test["label"].to_numpy(dtype=bool)
        label = f"{a.number} (`{_short(a.local_model)}`)"
        base = classification(y, np.ones(len(y)))
        lines.append(
            f"| {label} | always says *right* | {_pct(base['accuracy'])} | "
            f"{_pct(base['precision'])} | 100.0% | {base['f1']:.3f} | 100.0% |"
        )
        for name in ("heuristic", "random", *LEARNED):
            m = classification(y, test[f"score_{name}"].to_numpy(dtype=float))
            lines.append(
                f"| {label} | {name} | {_pct(m['accuracy'])} | {_pct(m['precision'])} | "
                f"{_pct(m['recall'])} | {m['f1']:.3f} | {_pct(m['predicted_right'])} |"
            )
    lines += [
        "",
        "A router is only useful if its precision clearly beats the *always says right* row (the "
        "base rate). Routing decisions in the cost analysis use the threshold sweep, not 0.5.",
    ]
    return "\n".join(lines)


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
    write_into_report(readme, setup_table(settings), *_block("readme-setup"))
    write_into_report(
        readme, classification_table(attempts, settings), *_block("readme-classification")
    )
    names, acc = model_accuracy(settings)
    write_into_report(readme, benchmark_table(settings, names, acc), *_block("readme-benchmarks"))
    plot_benchmarks(
        settings, names, acc, settings.reports_dir / "figures" / "benchmark-accuracy.png"
    )
    for a in attempts:
        figure = settings.reports_dir / "figures" / f"reliability-attempt{a.number}.png"
        scores = pd.read_parquet(settings.results_dir / f"router-attempt{a.number}-scores.parquet")
        plot_reliability(scores, figure, f"Attempt {a.number}, {_short(a.local_model)}")
    log.info("readme_done", readme=str(readme), attempts=list(SHOWN))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
