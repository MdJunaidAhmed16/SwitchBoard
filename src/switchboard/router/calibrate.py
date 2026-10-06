"""Temperature scaling (05-router-model, Calibration).

Fine-tuned classifiers are overconfident. A single temperature T rescales the logits,
p = sigmoid(z / T), fitted to minimise log loss on the **validation** set only. It changes
confidence, never the ranking, so AUROC is unaffected and ECE / Brier should improve.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sigmoid(z: NDArray[np.float64]) -> NDArray[np.float64]:
    out: NDArray[np.float64] = 1.0 / (1.0 + np.exp(-z))
    return out


def _log_loss(z: NDArray[np.float64], y: NDArray[np.float64], t: float) -> float:
    p = np.clip(sigmoid(z / t), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(logits: ArrayLike, labels: ArrayLike) -> float:
    """The T in [0.05, 20] minimising validation log loss: coarse log-grid, then a fine grid."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    best = min(grid, key=lambda t: _log_loss(z, y, float(t)))
    fine = np.linspace(best * 0.95, best * 1.05, 101)
    return float(min(fine, key=lambda t: _log_loss(z, y, float(t))))


def apply_temperature(logits: ArrayLike, t: float) -> NDArray[np.float64]:
    return sigmoid(np.asarray(logits, dtype=np.float64) / t)
