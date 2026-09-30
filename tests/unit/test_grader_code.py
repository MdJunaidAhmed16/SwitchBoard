"""Code graders. Fixtures are hand-written and trusted, so they run in a host subprocess here;
the same assembly is executed inside Docker by tests/unit/test_sandbox.py (marker: docker)."""

import json

import pytest

from switchboard.labeling.graders import grader_for
from switchboard.labeling.graders.code import (
    CodeGrader,
    build_humaneval_program,
    build_mbpp_program,
    extract_code,
)
from switchboard.labeling.sandbox import TrustedSubprocessSandbox

HUMANEVAL_REF = json.dumps(
    {
        "prompt": (
            "from typing import List\n\n\n"
            "def add_all(xs: List[int]) -> int:\n"
            '    """Return the sum of xs.\n    >>> add_all([1, 2])\n    3\n    """\n'
        ),
        "test": (
            "def check(candidate):\n"
            "    assert candidate([1, 2]) == 3\n"
            "    assert candidate([]) == 0\n"
            "    assert candidate([-1, 1, 5]) == 5\n"
        ),
        "entry_point": "add_all",
    }
)

MBPP_REF = json.dumps(
    {
        "test_setup_code": "",
        "test_list": [
            "assert is_even(2) == True",
            "assert is_even(3) == False",
            "assert is_even(0) == True",
        ],
    }
)

FULL_FUNCTION = (
    "Here is the implementation:\n\n```python\n"
    "def add_all(xs: List[int]) -> int:\n    return sum(xs)\n```\n"
)


@pytest.fixture
def humaneval() -> CodeGrader:
    return CodeGrader("humaneval", TrustedSubprocessSandbox(timeout_s=5))


@pytest.fixture
def mbpp() -> CodeGrader:
    return CodeGrader("mbpp", TrustedSubprocessSandbox(timeout_s=5))


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        # Full function restated, relies on the prompt's `from typing import List`.
        (FULL_FUNCTION, True),
        # Continuation of the stub (body only).
        ("```python\n    return sum(xs)\n```", True),
        # No fences at all, but a complete function.
        (
            "def add_all(xs):\n    total = 0\n    for x in xs:\n        total += x\n"
            "    return total",
            True,
        ),
        # Prose, then a non-code block, then the code block.
        (
            "Plan:\n```\nloop and add\n```\n```python\ndef add_all(xs):\n    return sum(xs)\n```",
            True,
        ),
        # Unterminated fence (generation hit the token cap) that still holds working code.
        ("```python\ndef add_all(xs):\n    return sum(xs)\n", True),
        # Wrong logic.
        ("```python\ndef add_all(xs):\n    return sum(xs) + 1\n```", False),
        # Off-by-one that passes the doctest but not the hidden tests.
        ("```python\ndef add_all(xs):\n    return xs[0] + xs[1]\n```", False),
        # Syntax error.
        ("```python\ndef add_all(xs)\n    return sum(xs)\n```", False),
        # Wrong function name.
        ("```python\ndef total(xs):\n    return sum(xs)\n```", False),
        # Raises at import time.
        ("```python\nraise SystemExit(3)\n```", False),
        # Empty and refusal.
        ("", False),
        ("I'm sorry, but I can't write that code.", False),
    ],
)
def test_humaneval_fixtures(humaneval: CodeGrader, prediction: str, expected: bool) -> None:
    assert humaneval(prediction, HUMANEVAL_REF) is expected


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ("```python\ndef is_even(n):\n    return n % 2 == 0\n```", True),
        ("```py\ndef is_even(n):\n    return not n & 1\n```", True),
        (
            "Sure!\n```python\ndef is_even(n):\n    if n % 2 == 0:\n        return True\n"
            "    return False\n```\nThis works.",
            True,
        ),
        ("def is_even(n):\n    return n % 2 == 0", True),
        ("```python\nimport math\n\ndef is_even(n):\n    return math.fmod(n, 2) == 0\n```", True),
        ("```python\ndef is_even(n):\n    return n % 2 == 1\n```", False),
        ("```python\ndef is_even(n):\n    return True\n```", False),
        ("```python\ndef even(n):\n    return n % 2 == 0\n```", False),
        ("```python\ndef is_even(n):\n    retrun n % 2 == 0\n```", False),
        ("```python\ndef is_even(n):\n    return None\n```", False),
        ("", False),
        ("I cannot help with this.", False),
    ],
)
def test_mbpp_fixtures(mbpp: CodeGrader, prediction: str, expected: bool) -> None:
    assert mbpp(prediction, MBPP_REF) is expected


def test_infinite_loop_is_killed_and_fails() -> None:
    grader = CodeGrader("mbpp", TrustedSubprocessSandbox(timeout_s=1))
    assert grader("```python\ndef is_even(n):\n    while True: pass\n```", MBPP_REF) is False


def test_mbpp_setup_code_runs_before_tests() -> None:
    ref = json.dumps({"test_setup_code": "BASE = 10", "test_list": ["assert f() == 10"]})
    program = build_mbpp_program("```python\ndef f():\n    return BASE\n```", ref)
    assert program is not None
    assert program.index("BASE = 10") < program.index("assert f()")


def test_humaneval_program_calls_check_on_entry_point() -> None:
    program = build_humaneval_program(FULL_FUNCTION, HUMANEVAL_REF)
    assert program is not None
    assert program.rstrip().endswith("check(add_all)")


def test_extract_prefers_block_with_def() -> None:
    text = "```python\nprint('demo')\n```\n```python\ndef g():\n    pass\n```"
    assert extract_code(text).startswith("def g")


def test_code_benchmarks_require_a_sandbox() -> None:
    with pytest.raises(ValueError, match="sandbox"):
        grader_for("humaneval")
