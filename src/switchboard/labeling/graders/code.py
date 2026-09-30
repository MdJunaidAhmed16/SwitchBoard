"""HumanEval and MBPP: pass iff the unit tests exit cleanly inside the sandbox.

References are JSON strings so the grader keeps the ``(prediction, reference) -> bool`` shape:

- HumanEval: ``{"prompt": ..., "test": ..., "entry_point": ...}``
- MBPP:      ``{"test_setup_code": ..., "test_list": [...]}``

Program assembly is pure and tested on its own; execution is delegated to a ``Sandbox``.
"""

from __future__ import annotations

import json
import re

from switchboard.labeling.sandbox import Sandbox

_FENCE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(prediction: str) -> str:
    """The first fenced block that defines a function, else the first block, else the raw text."""
    blocks = _FENCE.findall(prediction)
    if not blocks:
        # An unterminated fence (generation hit the token cap) still carries code.
        opened = re.search(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*)", prediction, re.DOTALL)
        return opened.group(1) if opened else prediction
    for block in blocks:
        if re.search(r"^\s*def\s", block, re.MULTILINE):
            return str(block)
    return str(blocks[0])


def _header_imports(prompt: str) -> str:
    lines = [ln for ln in prompt.splitlines() if ln.startswith(("import ", "from "))]
    return "\n".join(lines)


def build_humaneval_program(prediction: str, reference: str) -> str | None:
    ref = json.loads(reference)
    code = extract_code(prediction)
    if not code.strip():
        return None
    entry = ref["entry_point"]
    if re.search(rf"^\s*def\s+{re.escape(entry)}\s*\(", code, re.MULTILINE):
        body = _header_imports(ref["prompt"]) + "\n\n" + code
    else:
        # The model continued the stub instead of restating it.
        body = ref["prompt"] + code
    return f"{body}\n\n{ref['test']}\n\ncheck({entry})\n"


def build_mbpp_program(prediction: str, reference: str) -> str | None:
    ref = json.loads(reference)
    code = extract_code(prediction)
    if not code.strip():
        return None
    tests = "\n".join(ref["test_list"])
    return f"{code}\n\n{ref.get('test_setup_code') or ''}\n\n{tests}\n"


class CodeGrader:
    def __init__(self, benchmark: str, sandbox: Sandbox) -> None:
        builders = {"humaneval": build_humaneval_program, "mbpp": build_mbpp_program}
        self._build = builders[benchmark]
        self._sandbox = sandbox

    def __call__(self, prediction: str, reference: str) -> bool:
        program = self._build(prediction, reference)
        if program is None:
            return False
        return self._sandbox.run(program).passed
