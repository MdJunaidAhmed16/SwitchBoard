"""One grader per benchmark. Every grader has the shape ``(prediction, reference) -> bool``."""

from __future__ import annotations

from collections.abc import Callable

from switchboard.labeling.graders.bbh import grade_bbh
from switchboard.labeling.graders.choice import grade_choice
from switchboard.labeling.graders.code import CodeGrader
from switchboard.labeling.graders.math import grade_math
from switchboard.labeling.graders.numeric import grade_gsm8k
from switchboard.labeling.sandbox import Sandbox

Grader = Callable[[str, str], bool]

NUMERIC_BENCHMARKS = frozenset({"gsm8k", "gsm_hard", "svamp"})
CHOICE_BENCHMARKS = frozenset(
    {"mmlu", "arc_challenge", "aqua_rat", "commonsense_qa", "medmcqa", "qasc"}
)

_PURE: dict[str, Grader] = {
    **dict.fromkeys(NUMERIC_BENCHMARKS, grade_gsm8k),
    **dict.fromkeys(CHOICE_BENCHMARKS, grade_choice),
    "math": grade_math,
    "bbh": grade_bbh,
}
CODE_BENCHMARKS = frozenset({"humaneval", "mbpp"})


def grader_for(benchmark: str, sandbox: Sandbox | None = None) -> Grader:
    if benchmark in _PURE:
        return _PURE[benchmark]
    if benchmark in CODE_BENCHMARKS:
        if sandbox is None:
            raise ValueError(f"{benchmark} executes model code and needs a sandbox")
        return CodeGrader(benchmark, sandbox)
    raise KeyError(f"no grader for benchmark {benchmark!r}")


__all__ = ["CHOICE_BENCHMARKS", "CODE_BENCHMARKS", "NUMERIC_BENCHMARKS", "Grader", "grader_for"]
