"""Grader self-check over every real benchmark item, before any GPU time is spent.

For each item an *oracle* prediction is built from the reference (or, for code, the benchmark's
canonical solution) and must grade True. Where a wrong answer is easy to construct, it must grade
False. A failure means the grader and the dataset's reference format disagree, which would
silently corrupt every label for that benchmark.

    python -m switchboard.labeling.selfcheck --bench all
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

from switchboard.config import Settings, get_settings
from switchboard.labeling.benchmarks import REGISTRY, Item, load_items, load_raw
from switchboard.labeling.graders import (
    CHOICE_BENCHMARKS,
    CODE_BENCHMARKS,
    NUMERIC_BENCHMARKS,
    grader_for,
)
from switchboard.labeling.sandbox import DockerSandbox
from switchboard.log import configure_logging, get_logger

log = get_logger(__name__)


def _canonical_code(name: str, settings: Settings) -> dict[str, str]:
    raw = load_raw(REGISTRY[name], settings)
    if name == "humaneval":
        return {
            str(t): f"```python\n{p}{s}```"
            for t, p, s in zip(
                raw["task_id"], raw["prompt"], raw["canonical_solution"], strict=True
            )
        }
    return {
        f"mbpp-{int(t):04d}": f"```python\n{c}\n```"
        for t, c in zip(raw["task_id"], raw["code"], strict=True)
    }


def oracle(item: Item) -> tuple[str, str | None]:
    """(prediction that must pass, prediction that must fail or None)."""
    ref = item.reference
    if item.benchmark in NUMERIC_BENCHMARKS:
        return f"Working...\nThe answer is: {ref}", f"The answer is: {Decimal(ref) + 1}"
    if item.benchmark == "math":
        return f"So the answer is $\\boxed{{{ref}}}$.", None
    if item.benchmark in CHOICE_BENCHMARKS:
        wrong = "B" if ref == "A" else "A"
        return f"Reasoning.\nThe answer is ({ref})", f"The answer is ({wrong})"
    if item.benchmark == "bbh":
        return f"Let's think step by step.\nThe answer is {ref}.", None
    raise KeyError(item.benchmark)


def check(name: str, settings: Settings, sandbox: DockerSandbox | None) -> int:
    items = load_items(name, settings)
    grader = grader_for(name, sandbox)
    failures: list[str] = []
    if name in CODE_BENCHMARKS:
        solutions = _canonical_code(name, settings)
        with ThreadPoolExecutor(max_workers=settings.sandbox_workers) as pool:
            results = list(pool.map(lambda it: grader(solutions[it.item_id], it.reference), items))
        failures = [it.item_id for it, ok in zip(items, results, strict=True) if not ok]
    else:
        for item in items:
            good, bad = oracle(item)
            if not grader(good, item.reference):
                failures.append(f"{item.item_id}: oracle failed (ref={item.reference!r})")
            if bad is not None and grader(bad, item.reference):
                failures.append(f"{item.item_id}: wrong answer passed (ref={item.reference!r})")
    log.info("selfcheck", benchmark=name, items=len(items), failures=len(failures))
    for failure in failures[:20]:
        log.warning("selfcheck_failure", benchmark=name, detail=failure)
    return len(failures)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="all", choices=["all", "pure", *REGISTRY])
    args = parser.parse_args(argv)
    configure_logging()
    settings = get_settings()
    if args.bench == "all":
        names = list(REGISTRY)
    elif args.bench == "pure":
        names = [n for n in REGISTRY if n not in CODE_BENCHMARKS]
    else:
        names = [args.bench]
    sandbox = DockerSandbox.from_settings(settings) if CODE_BENCHMARKS & set(names) else None
    total = sum(check(name, settings, sandbox) for name in names)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
