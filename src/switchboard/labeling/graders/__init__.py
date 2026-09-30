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

_PURE: dict[str, Grader] = {
    "gsm8k": grade_gsm8k,
    "math": grade_math,
    "mmlu": grade_choice,
    "arc_challenge": grade_choice,
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


__all__ = ["CODE_BENCHMARKS", "Grader", "grader_for"]
