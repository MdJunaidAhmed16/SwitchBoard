"""The router experiments, each fixed in code so any of them can be re-run exactly.

Attempt 1 is the original Phase 2 run and failed the kill gate. Attempt 2 is the design
pre-registered in ``reports/R2-kill-gate.md`` on 2026-10-05, before any attempt-2 result existed.
Both are kept: attempt 2 is reported beside attempt 1, never instead of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Attempt:
    number: int
    # Training benchmarks; None means every benchmark with the train role in data/splits.yaml.
    train_benchmarks: tuple[str, ...] | None
    # Benchmark-balanced sample weights (see balanced_weights).
    balanced: bool
    # Also fine-tune and evaluate v1.
    with_v1: bool

    @property
    def stem(self) -> str:
        return "router-v0" if self.number == 1 else f"router-attempt{self.number}"


ATTEMPTS: dict[int, Attempt] = {
    1: Attempt(number=1, train_benchmarks=("gsm8k", "mmlu", "mbpp"), balanced=False, with_v1=False),
    2: Attempt(number=2, train_benchmarks=None, balanced=True, with_v1=True),
}

V1_SEEDS: tuple[int, ...] = (0, 1, 2)


def balanced_weights(train: pd.DataFrame) -> pd.Series:
    """Weights under which knowing a prompt's benchmark tells you nothing about its label.

    Every benchmark gets the same total weight, split equally between its correct and its
    incorrect examples, so each benchmark is effectively 50/50. The only signal left to learn is
    within-benchmark difficulty. Weights are scaled to average 1.
    """
    n_benchmarks = train["benchmark"].nunique()
    counts = train.groupby(["benchmark", "label"])["label"].transform("size")
    classes = train.groupby("benchmark")["label"].transform("nunique")
    raw = 1.0 / (n_benchmarks * classes * counts)
    return (raw * len(train) / raw.sum()).rename("weight")
