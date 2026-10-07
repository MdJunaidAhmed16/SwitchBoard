"""Cost model and threshold sweep (13-testing-and-reports, Phase 2)."""

from itertools import pairwise

import numpy as np
import pytest

from switchboard.bench.cost import Prices
from switchboard.bench.sweep import (
    EvalSet,
    analyse,
    cost_at_retention,
    criterion_1,
    criterion_2,
    quality_at_escalation,
    sweep,
)


def test_cost_matches_a_hand_computed_value() -> None:
    # 1,200 prompt tokens at $4/M + 300 output tokens at $20/M = 0.0048 + 0.006 = $0.0108.
    assert Prices(4.0, 20.0).cost(1200, 300) == pytest.approx(0.0108)
    assert Prices(0.10, 0.20).cost([1000, 0], [0, 1000]).tolist() == pytest.approx([1e-4, 2e-4])


def _ev(n: int = 400, seed: int = 0) -> EvalSet:
    rng = np.random.default_rng(seed)
    difficulty = rng.random(n)
    local_ok = rng.random(n) > difficulty  # harder prompts fail more often
    return EvalSet(
        benchmark=np.array(["a"] * (n // 2) + ["b"] * (n - n // 2)),
        local_ok=local_ok,
        frontier_ok=rng.random(n) > 0.05,
        local_cost=np.full(n, 0.0001),
        frontier_cost=np.full(n, 0.01),
        scores={
            "heuristic": np.clip(1 - difficulty + rng.normal(0, 0.4, n), 0, 1),
            "random": rng.random(n),
            "good": np.clip(1 - difficulty + rng.normal(0, 0.1, n), 0, 1),
        },
    )


def test_escalation_falls_monotonically_as_tau_falls() -> None:
    curve = sweep(_ev(), _ev().scores["good"])
    rates = curve.sort_values("tau", ascending=False)["escalation_rate"].tolist()
    assert all(a >= b for a, b in pairwise(rates))


def test_sweep_endpoints_are_always_local_and_nearly_always_frontier() -> None:
    ev = _ev()
    curve = sweep(ev, ev.scores["good"])
    first = curve.iloc[0]  # tau 0: every score >= 0 → all local
    assert first["escalation_rate"] == 0.0
    assert first["quality"] == pytest.approx(ev.local_ok.mean())
    assert first["cost"] == pytest.approx(ev.local_cost.sum())


def test_sweep_is_reproducible() -> None:
    a = sweep(_ev(), _ev().scores["good"])
    b = sweep(_ev(), _ev().scores["good"])
    assert a.equals(b)


def test_cost_at_retention_and_quality_at_escalation() -> None:
    ev = _ev()
    curve = sweep(ev, ev.scores["good"])
    fq = float(ev.frontier_ok.mean())
    assert cost_at_retention(curve, fq, 0.0) == pytest.approx(curve["cost"].min())
    assert np.isnan(cost_at_retention(curve, fq, 5.0))
    assert quality_at_escalation(curve, 0.0) == pytest.approx(ev.local_ok.mean())


def test_a_better_router_dominates_and_beats_random() -> None:
    ev = _ev()
    fq = float(ev.frontier_ok.mean())
    good, heur, rand = (sweep(ev, ev.scores[k]) for k in ("good", "heuristic", "random"))
    assert criterion_1(good, heur, fq)["dominates"]
    assert criterion_2(good, rand)["beats_random"]
    assert not criterion_2(rand, rand)["beats_random"]  # strictly better is required


def test_analyse_marks_only_learned_routers_for_the_gate() -> None:
    result = analyse(_ev(), with_bootstrap=False)
    assert "criterion_1" in result["routers"]["good"]
    assert "criterion_1" not in result["routers"]["heuristic"]
    assert "criterion_1" not in result["routers"]["random"]
    assert result["always_frontier"]["cost"] == pytest.approx(0.01 * 400)
