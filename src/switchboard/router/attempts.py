"""The router experiments, each fixed in code so any of them can be re-run exactly.

Attempt 1 is the original Phase 2 run and failed the kill gate. Attempt 2 is the design
pre-registered in ``reports/R2-kill-gate.md`` on 2026-10-05, before any attempt-2 result existed.
Attempt 3 is attempt 2 unchanged except for the local model (Qwen2.5-7B-Instruct, AWQ 4-bit),
pre-registered in the same report on 2026-10-08. Every attempt is kept and reported beside the
others, never instead of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from switchboard.config import Settings


@dataclass(frozen=True)
class Attempt:
    number: int
    # Training benchmarks; None means every benchmark with the train role in data/splits.yaml.
    train_benchmarks: tuple[str, ...] | None
    # Benchmark-balanced sample weights (see balanced_weights).
    balanced: bool
    # Also fine-tune and evaluate v1.
    with_v1: bool
    # (model id, revision) of the local model whose labels are learned; None means the default
    # in config.py.
    local_model: tuple[str, str] | None = None

    @property
    def stem(self) -> str:
        return "router-v0" if self.number == 1 else f"router-attempt{self.number}"

    def settings_for(self, settings: Settings) -> Settings:
        """``settings`` with this attempt's local model, so its labels file is the one read."""
        if self.local_model is None:
            return settings
        model_id, revision = self.local_model
        return settings.model_copy(
            update={"local_model_id": model_id, "local_model_revision": revision}
        )


ATTEMPTS: dict[int, Attempt] = {
    1: Attempt(number=1, train_benchmarks=("gsm8k", "mmlu", "mbpp"), balanced=False, with_v1=False),
    2: Attempt(number=2, train_benchmarks=None, balanced=True, with_v1=True),
    3: Attempt(
        number=3,
        train_benchmarks=None,
        balanced=True,
        with_v1=True,
        local_model=("Qwen/Qwen2.5-7B-Instruct-AWQ", "b25037543e9394b818fdfca67ab2a00ecc7dd641"),
    ),
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
