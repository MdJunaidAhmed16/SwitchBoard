"""README generator (12-demo-and-readme): numbers come from results, never typed by hand."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from switchboard.bench.readme import EN_DASH, LEARNED, headline, passes, reliability_curve
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


def _sweep(c1: tuple[bool, ...], c2: tuple[bool, ...], ci: tuple[float, float]) -> dict[str, Any]:
    routers = {
        "heuristic": _router(0.114, 0.918, False, False, (0, 0)),
        "random": _router(0.079, 0.930, False, False, (0, 0)),
    }
    for i, name in enumerate(LEARNED):
        routers[name] = _router(0.07 + 0.01 * i, 0.94 - 0.01 * i, c1[i], c2[i], ci)
    return {"routers": routers}


def test_passes_reads_both_criteria() -> None:
    assert passes(_router(0.1, 0.9, True, False, (0, 0))) == (True, False)


def test_headline_reports_a_failed_gate_and_v1_as_a_range() -> None:
    text = headline(_sweep((True, False, False, False), (False, True, False, False), (-1.0, 1.0)))
    assert "kill gate failed" in text
    assert "every 95% bootstrap interval includes zero" in text
    # v1 seeds keep 8%, 9%, 10% local: reported as a range, never a cherry-picked seed.
    assert f"8.0%{EN_DASH}10.0%" in text
    assert "11.4%" in text
    assert "93.0%" in text


def test_headline_reports_a_pass_when_one_router_passes_both() -> None:
    text = headline(_sweep((False, True, False, False), (False, True, False, False), (-1.0, -0.1)))
    assert "kill gate passed" in text
    assert "includes zero" not in text


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
