"""Draw a random sample of (prompt, generation, label) triples for the Phase 1 manual gate.

The gate (13-testing-and-reports): inspect 30 random triples; more than one mislabel means the
grader is fixed and labels are re-graded. The sheet is written to ``reports/review/R1-sample.md``.

    python -m switchboard.labeling.review --n 30 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence

import pandas as pd

from switchboard.config import get_settings
from switchboard.labeling.generate import labels_path
from switchboard.log import configure_logging, get_logger

log = get_logger(__name__)


def sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    order = df.apply(
        lambda r: hashlib.sha256(f"{seed}:{r['benchmark']}:{r['item_id']}".encode()).hexdigest(),
        axis=1,
    )
    return df.assign(_order=order).sort_values("_order").head(n).drop(columns="_order")


def _fence(text: str) -> str:
    return "````text\n" + text.replace("````", "` ` ` `") + "\n````"


def render(rows: pd.DataFrame, seed: int) -> str:
    out = [
        "# R1 manual label review",
        "",
        f"Sample: {len(rows)} random triples, seed {seed}. "
        "Gate: more than 1 mislabel → fix the grader, re-grade, re-draw.",
        "",
        "- Reviewer: ",
        "- Date: ",
        "- Mislabels found: ",
        "",
        "Mark each item: `[x] label correct` or `[x] MISLABELLED` with a one-line reason.",
        "",
    ]
    for i, r in enumerate(rows.itertuples(index=False), start=1):
        verdict = "CORRECT" if r.label else "WRONG"
        out += [
            f"## {i}. {r.benchmark} / {r.item_id} — grader said **{verdict}**",
            "",
            "**Prompt sent to the model**",
            "",
            _fence(str(r.user_prompt)),
            "",
            "**Reference**",
            "",
            _fence(str(r.reference)),
            "",
            f"**Generation** (finish: {r.finish_reason}, {r.output_tokens} tokens)",
            "",
            _fence(str(r.generation)),
            "",
            "- [ ] label correct",
            "- [ ] MISLABELLED — reason: ",
            "",
        ]
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    configure_logging()
    settings = get_settings()
    df = pd.read_parquet(labels_path(settings))
    path = settings.reports_dir / "review" / "R1-sample.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(sample(df, args.n, args.seed), args.seed))
    log.info("review_sheet_written", path=str(path), n=args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
