"""Routers, metrics and the shared interface (13-testing-and-reports, Phase 2)."""

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from switchboard.router.baselines import HeuristicRouter, RandomRouter, heuristic_features
from switchboard.router.interface import Router
from switchboard.router.metrics import (
    at_threshold,
    auroc,
    bootstrap_ci,
    brier,
    ece,
    paired_auroc_difference_ci,
)
from switchboard.router.v0 import V0Router


class KeywordEmbedder:
    """A deterministic stand-in for the encoder: 'hard' in the text is the only signal."""

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return np.array([[1.0 if "hard" in t else 0.0, len(t) / 100] for t in texts], np.float32)


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hard = rng.random(n) < 0.4
    texts = [f"{'hard' if h else 'easy'} question number {i}" for i, h in enumerate(hard)]
    return pd.DataFrame(
        {"router_text": texts, "label": ~hard, "benchmark": "b", "item_id": range(n)}
    )


def _routers() -> list[Router]:
    return [HeuristicRouter(), RandomRouter(seed=0), V0Router(KeywordEmbedder())]


# --- Interface parity ----------------------------------------------------------------------------


@pytest.mark.parametrize("router", _routers(), ids=lambda r: r.name)
def test_every_router_shares_the_interface(router: Router) -> None:
    train, val = _frame(seed=0), _frame(seed=1)
    assert isinstance(router, Router)
    router.fit(train, val)
    p = router.predict_proba(list(val["router_text"]))
    assert p.shape == (len(val),)
    assert p.dtype == np.float64
    assert np.all((p >= 0) & (p <= 1))
    # Scores depend only on the text, never on batch composition or order.
    reversed_p = router.predict_proba(list(val["router_text"])[::-1])
    assert np.allclose(p, reversed_p[::-1])


def test_router_names_are_distinct() -> None:
    assert len({r.name for r in _routers()}) == 3


# --- Baselines ----------------------------------------------------------------------------------


def test_heuristic_features_are_length_and_keyword_count() -> None:
    f = heuristic_features(["Prove that the polynomial has an integer root.", "Hi"])
    assert f.shape == (2, 2)
    assert f[0, 1] == 3  # prove, polynomial, integer
    assert f[1, 1] == 0
    assert f[0, 0] > f[1, 0]


def test_keyword_match_is_whole_word() -> None:
    assert heuristic_features(["improve the stepwise output"])[0, 1] == 0


def test_random_router_ignores_content_and_is_uniform() -> None:
    texts = [f"prompt {i}" for i in range(5000)]
    p = RandomRouter(seed=0).predict_proba(texts)
    assert 0.48 < p.mean() < 0.52
    assert not np.allclose(p, RandomRouter(seed=1).predict_proba(texts))
    assert np.allclose(p, RandomRouter(seed=0).predict_proba(texts))


def test_v0_chooses_c_on_validation_and_separates_signal() -> None:
    router = V0Router(KeywordEmbedder())
    train, val = _frame(seed=0), _frame(seed=1)
    router.fit(train, val)
    assert router.chosen_c in router.c_grid
    assert set(router.val_auroc_by_c) == set(router.c_grid)
    assert auroc(val["label"], router.predict_proba(list(val["router_text"]))) == 1.0


def test_v0_refuses_to_predict_before_fit() -> None:
    with pytest.raises(RuntimeError, match="before fit"):
        V0Router(KeywordEmbedder()).predict_proba(["x"])


# --- Metrics ------------------------------------------------------------------------------------


def test_auroc_known_values() -> None:
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_ece_is_zero_for_perfect_calibration_and_large_when_overconfident() -> None:
    y = np.array([1] * 70 + [0] * 30)
    assert ece(y, np.full(100, 0.7)) == pytest.approx(0.0, abs=1e-9)
    assert ece(y, np.full(100, 0.99)) == pytest.approx(0.29, abs=1e-9)


def test_brier_known_value() -> None:
    assert brier([1, 0], [0.75, 0.25]) == pytest.approx(0.0625)


def test_bootstrap_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    lo, hi = bootstrap_ci(y, p, n_boot=300)
    assert lo < auroc(y, p) < hi


def test_paired_difference_of_a_router_with_itself_is_zero() -> None:
    rng = np.random.default_rng(1)
    y, p = rng.integers(0, 2, 300), rng.random(300)
    diff, lo, hi = paired_auroc_difference_ci(y, p, p, n_boot=200)
    assert diff == 0.0
    assert lo == hi == 0.0


def test_at_threshold_counts_the_four_outcomes() -> None:
    # p >= tau routes local. Two routed local (one right, one wrong), two escalated.
    r = at_threshold(y=[1, 0, 1, 0], p=[0.9, 0.8, 0.3, 0.1], tau=0.5)
    assert (r.saved, r.wrong_shipped, r.wasted, r.escalated_correctly) == (1, 1, 1, 1)
    assert r.escalation_rate == 0.5
    assert r.accuracy == 0.5


def test_escalation_rate_falls_as_tau_falls() -> None:
    rng = np.random.default_rng(2)
    y, p = rng.integers(0, 2, 400), rng.random(400)
    rates = [at_threshold(y, p, tau).escalation_rate for tau in np.linspace(1, 0, 101)]
    assert all(a >= b for a, b in pairwise(rates))
