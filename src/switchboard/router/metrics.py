"""Router metrics (05-router-model). Accuracy is never reported without the base rate beside it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.metrics import roc_auc_score


def _arrays(y: ArrayLike, p: ArrayLike) -> tuple[NDArray[np.int_], NDArray[np.float64]]:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=np.float64)
    if y_arr.shape != p_arr.shape:
        raise ValueError(f"shape mismatch: {y_arr.shape} vs {p_arr.shape}")
    return y_arr, p_arr


def base_rate(y: ArrayLike) -> float:
    return float(np.mean(np.asarray(y, dtype=float)))


def auroc(y: ArrayLike, p: ArrayLike) -> float:
    y_arr, p_arr = _arrays(y, p)
    return float(roc_auc_score(y_arr, p_arr))


def brier(y: ArrayLike, p: ArrayLike) -> float:
    y_arr, p_arr = _arrays(y, p)
    return float(np.mean((p_arr - y_arr) ** 2))


def ece(y: ArrayLike, p: ArrayLike, bins: int = 10) -> float:
    """Expected calibration error with equal-width bins (05-router-model: 10 bins)."""
    y_arr, p_arr = _arrays(y, p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(p_arr, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.mean() * abs(p_arr[mask].mean() - y_arr[mask].mean())
    return float(total)


def bootstrap_ci(
    y: ArrayLike, p: ArrayLike, n_boot: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap interval for AUROC. Resamples that contain one class are skipped."""
    y_arr, p_arr = _arrays(y, p)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if y_arr[idx].min() == y_arr[idx].max():
            continue
        stats.append(roc_auc_score(y_arr[idx], p_arr[idx]))
    lo, hi = np.percentile(stats, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def paired_auroc_difference_ci(
    y: ArrayLike, p_a: ArrayLike, p_b: ArrayLike, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """AUROC(a) - AUROC(b) with a paired bootstrap 95% interval: (difference, low, high)."""
    y_arr, a = _arrays(y, p_a)
    _, b = _arrays(y, p_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if y_arr[idx].min() == y_arr[idx].max():
            continue
        diffs.append(roc_auc_score(y_arr[idx], a[idx]) - roc_auc_score(y_arr[idx], b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return auroc(y_arr, a) - auroc(y_arr, b), float(lo), float(hi)


@dataclass(frozen=True)
class AtThreshold:
    """Routing outcome at one threshold τ: p ≥ τ routes local, p < τ escalates."""

    tau: float
    escalation_rate: float
    accuracy: float
    saved: int  # routed local, local correct
    wrong_shipped: int  # routed local, local wrong  ← the expensive error
    wasted: int  # escalated, local would have been correct
    escalated_correctly: int  # escalated, local wrong


def at_threshold(y: ArrayLike, p: ArrayLike, tau: float) -> AtThreshold:
    y_arr, p_arr = _arrays(y, p)
    local = p_arr >= tau
    return AtThreshold(
        tau=tau,
        escalation_rate=float(np.mean(~local)),
        accuracy=float(np.mean(local == y_arr.astype(bool))),
        saved=int(np.sum(local & (y_arr == 1))),
        wrong_shipped=int(np.sum(local & (y_arr == 0))),
        wasted=int(np.sum(~local & (y_arr == 1))),
        escalated_correctly=int(np.sum(~local & (y_arr == 0))),
    )
