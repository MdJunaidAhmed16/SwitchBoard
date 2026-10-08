"""README generator (12-demo-and-readme): numbers come from results, never typed by hand."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from switchboard.bench.readme import (
    EN_DASH,
    LEARNED,
    V1_SEEDS,
    AttemptResult,
    gate_passed,
    headline,
    passes,
    reliability_curve,
)
from switchboard.labeling.summary import write_into_report


def _router(
    kept: float, cost: float, c1: bool, c2: bool, ci: tuple[float, float]
) -> dict[str, Any]:
    return {
        "operating_point": {
            "found": True,
            "tau": 0.5,
            "share_kept_local": kept,
            "quality_retention": 0.95,
            "cost_share_of_always_frontier": cost,
        },
        "criterion_1": {"levels": [], "dominates": c1},
        "criterion_2": {"levels": [], "beats_random": c2},
        "cost_vs_heuristic_at_95": {"diff_usd": 0.0, "ci95": list(ci)},
    }


def _attempt(
    c1: tuple[bool, ...],
    c2: tuple[bool, ...],
    ci: tuple[float, float] = (-1.0, 1.0),
    c3: tuple[bool, bool] = (True, True),
) -> AttemptResult:
    routers = {
        "heuristic": _router(0.114, 0.918, False, False, (0, 0)),
        "random": _router(0.079, 0.930, False, False, (0, 0)),
    }
    for i, name in enumerate(LEARNED):
        routers[name] = _router(0.07 + 0.01 * i, 0.94 - 0.01 * i, c1[i], c2[i], ci)
    return AttemptResult(
        number=3,
        local_model="Org/Local-7B",
        router={"kill_gate": {"criterion_3_pass": c3[0], "criterion_3_v1_pass_all_seeds": c3[1]}},
        sweep={"routers": routers, "always_local": {"quality": 0.64}},
    )


ALL = (True, True, True, True)
NONE = (False, False, False, False)


def test_passes_reads_both_criteria() -> None:
    assert passes(_router(0.1, 0.9, True, False, (0, 0))) == (True, False)


def test_gate_needs_v0_or_every_v1_seed_to_pass_all_three_criteria() -> None:
    assert not gate_passed(_attempt(NONE, NONE))
    # One v1 seed passing criteria 1 and 2 is not enough: every seed must.
    assert not gate_passed(_attempt((False, True, False, False), (False, True, False, False)))
    assert gate_passed(_attempt((False, True, True, True), (False, True, True, True)))
    assert gate_passed(_attempt((True, False, False, False), (True, False, False, False)))
    # Criterion 3 still has to hold.
    assert not gate_passed(_attempt(ALL, ALL, c3=(False, False)))


def test_headline_reports_v1_as_a_range_and_the_gate() -> None:
    text = headline([_attempt(NONE, NONE)])
    assert "Local-7B" in text
    assert "64.0%" in text
    # v1 seeds keep 8%, 9%, 10% local: reported as a range, never a cherry-picked seed.
    assert f"8.0%{EN_DASH}10.0% kept local" in text
    assert "11.4% kept local" in text
    assert "**failed**" in text
    assert "| no seed |" in text


def test_headline_counts_seeds_significantly_cheaper_than_the_heuristic() -> None:
    text = headline([_attempt(ALL, ALL, ci=(-1.0, -0.1))])
    assert f"{len(V1_SEEDS)} of 3 seeds" in text
    assert "| passed |" in text


def test_reliability_curve_matches_hand_computed_bins() -> None:
    p = np.array([0.05, 0.05, 0.95, 0.95, 0.95, 0.55])
    y = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    mean_p, acc, count = reliability_curve(y, p)
    assert mean_p.tolist() == pytest.approx([0.05, 0.55, 0.95])
    assert acc.tolist() == pytest.approx([0.5, 1.0, 2 / 3])
    assert count.tolist() == [2, 1, 3]
    # Sparse bins are dropped, not merged.
    assert reliability_curve(y, p, min_count=2)[2].tolist() == [2, 3]


def test_generated_block_replaces_only_its_markers(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    begin, end = "<!-- BEGIN GENERATED: x -->", "<!-- END GENERATED: x -->"
    readme.write_text(f"intro\n{begin}\nold\n{end}\noutro\n")
    write_into_report(readme, "new", begin, end)
    write_into_report(readme, "new", begin, end)  # idempotent
    assert readme.read_text() == f"intro\n{begin}\nnew\n{end}\noutro\n"
