"""Per-benchmark label statistics → ``results/labels-summary.json`` and the R1 report tables.

Every number in ``reports/R1-labels.md`` between the GENERATED markers is written by this script
from the labels file. None is typed by hand.

    python -m switchboard.labeling.summary
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from switchboard.config import Settings, get_settings
from switchboard.labeling.benchmarks import REGISTRY
from switchboard.labeling.generate import labels_path, meta_path
from switchboard.labeling.schema import validate_labels
from switchboard.labeling.splits import load_splits
from switchboard.log import configure_logging, get_logger

log = get_logger(__name__)

# 13-testing-and-reports: a near-constant label teaches the router nothing.
FLAG_HIGH = 0.95
FLAG_LOW = 0.05

BEGIN = "<!-- BEGIN GENERATED: labels-summary -->"
END = "<!-- END GENERATED: labels-summary -->"


def summarise(df: pd.DataFrame, roles: dict[str, str], meta: dict[str, Any]) -> dict[str, Any]:
    validate_labels(df)
    wall: dict[str, float] = {}
    for run in meta.get("runs", []):
        wall[run["benchmark"]] = wall.get(run["benchmark"], 0.0) + run["generation_wall_clock_s"]

    bench_meta = meta.get("benchmarks", {})
    per_benchmark = {}
    for name, group in df.groupby("benchmark", sort=True):
        rate = float(group["label"].mean())
        excluded = bench_meta.get(str(name), {}).get("excluded", {})
        per_benchmark[str(name)] = {
            "role": roles.get(str(name), "unassigned"),
            "n": len(group),
            "correct": int(group["label"].sum()),
            "accuracy": rate,
            "flag": "near-constant" if rate > FLAG_HIGH or rate < FLAG_LOW else None,
            "hit_token_cap_rate": float((group["finish_reason"] == "length").mean()),
            "mean_output_tokens": float(group["output_tokens"].mean()),
            "total_output_tokens": int(group["output_tokens"].sum()),
            "generation_wall_clock_s": wall.get(str(name)),
            "limited_run": bench_meta.get(str(name), {}).get("limited_run"),
            "excluded": len(excluded),
            "excluded_items": excluded,
        }

    by_role = {}
    for role in ("train", "val", "test"):
        subset = df[df["benchmark"].map(roles) == role]
        if len(subset):
            by_role[role] = {"n": len(subset), "base_rate": float(subset["label"].mean())}

    return {
        "model_id": meta.get("model_id"),
        "model_revision": meta.get("model_revision"),
        "decode_params": meta.get("decode_params"),
        "vllm": meta.get("vllm"),
        "software": meta.get("software"),
        "overall": {
            "n": len(df),
            "correct": int(df["label"].sum()),
            "base_rate": float(df["label"].mean()),
            "excluded": sum(b["excluded"] for b in per_benchmark.values()),
            "total_output_tokens": int(df["output_tokens"].sum()),
            "generation_wall_clock_s": round(sum(wall.values()), 1),
        },
        "by_role": by_role,
        "per_benchmark": per_benchmark,
        "missing_benchmarks": sorted(set(REGISTRY) - set(per_benchmark)),
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def to_markdown(summary: dict[str, Any]) -> str:
    o = summary["overall"]
    lines = [
        f"Model: `{summary['model_id']}` @ `{str(summary['model_revision'])[:10]}` · "
        f"decoding: `{json.dumps(summary['decode_params'], sort_keys=True)}`",
        "",
        f"**Overall base rate: {_pct(o['base_rate'])}** ({o['correct']} / {o['n']} correct). "
        "Every later accuracy is read against this.",
        "",
        "| Benchmark | Role | n | Correct | Accuracy | Excluded | Hit token cap "
        "| Mean out tokens | Flag |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, b in summary["per_benchmark"].items():
        flag = b["flag"] or ""
        if b["limited_run"]:
            flag = (flag + " limited run").strip()
        lines.append(
            f"| {name} | {b['role']} | {b['n']} | {b['correct']} | {_pct(b['accuracy'])} | "
            f"{b['excluded']} | {_pct(b['hit_token_cap_rate'])} | "
            f"{b['mean_output_tokens']:.0f} | {flag} |"
        )
    lines += ["", "| Role | n | Base rate |", "| --- | ---: | ---: |"]
    for role, r in summary["by_role"].items():
        lines.append(f"| {role} | {r['n']} | {_pct(r['base_rate'])} |")
    wall_h = o["generation_wall_clock_s"] / 3600
    lines += [
        "",
        f"Tokens generated: {o['total_output_tokens']:,} · generation wall-clock: {wall_h:.2f} h",
    ]
    excluded = [
        (name, item_id, reason)
        for name, b in summary["per_benchmark"].items()
        for item_id, reason in b["excluded_items"].items()
    ]
    if excluded:
        lines += ["", f"**Excluded items ({len(excluded)})** — never generated or labelled:"]
        lines += [f"- {name} / `{item_id}`: {reason}" for name, item_id, reason in excluded]
    if summary["missing_benchmarks"]:
        lines.append(f"\n**Not yet labelled:** {', '.join(summary['missing_benchmarks'])}")
    return "\n".join(lines)


def write_into_report(report: Path, block: str) -> None:
    text = report.read_text() if report.exists() else f"{BEGIN}\n{END}\n"
    if BEGIN not in text:
        text += f"\n{BEGIN}\n{END}\n"
    pattern = re.compile(re.escape(BEGIN) + ".*?" + re.escape(END), re.DOTALL)
    report.write_text(pattern.sub(lambda _: f"{BEGIN}\n{block}\n{END}", text))


def run(settings: Settings) -> dict[str, Any]:
    df = pd.read_parquet(labels_path(settings))
    meta = json.loads(meta_path(settings).read_text())
    roles = load_splits(settings.splits_path, known=set(REGISTRY))
    summary = summarise(df, dict(roles), meta)
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out = settings.results_dir / "labels-summary.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_into_report(settings.reports_dir / "R1-labels.md", to_markdown(summary))
    log.info("labels_summary_written", path=str(out), base_rate=summary["overall"]["base_rate"])
    return summary


def main() -> int:
    configure_logging()
    run(get_settings())
    return 0


if __name__ == "__main__":
    sys.exit(main())
