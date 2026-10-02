"""Multiple choice (MMLU, ARC): extract a single option letter.

Accepted, in order of precedence:

1. An explicit marker: "The answer is (B)", "Answer: C", "the correct choice is C",
   "the best option is D".
2. A boxed letter: ``\\boxed{B}`` or ``\\boxed{\\text{B}}``.
3. A concluding option line: the last line that starts with an option ("**B. No, unless ...**"),
   accepted only when the line before it ends with a colon ("... would be:"). An enumeration of
   options with no conclusion yields nothing, because its last option line follows another option
   or its commentary, not a colon.
4. The whole reply is a bare letter ("B", "(C)"), or starts with one ("D. Mitochondria ...").

Letters must be uppercase throughout: "the answer is a function" must not read as option A.
"""

from __future__ import annotations

import re

from switchboard.labeling.graders.common import extract_boxed

_MARKER = (
    r"(?i:(?:answer|correct\s+(?:choice|option)|best\s+(?:choice|option))"
    r"[\s*_]*(?:is)?[\s*_]*[:\-]?\s*(?:option\s*)?)"
)
_MARKED = re.compile(_MARKER + r"[\*\s]*\(?([A-J])\)?(?![A-Za-z])")
_BOXED_LETTER = re.compile(r"^\s*(?:\\text\{\s*)?\(?([A-J])\)?\s*\}?\s*$")
_OPTION_LINE = re.compile(r"^[\*_\s]*\(?([A-J])[\.\:\)]\s")
_BARE = re.compile(r"^[\*\s]*\(?([A-J])\)?[\.\:\)]?[\*\s]*$")
_LEADING = re.compile(r"^[\*\s]*\(?([A-J])[\.\:\)]\s")


def _concluding_option_line(text: str) -> str | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    option_lines = [i for i, line in enumerate(lines) if _OPTION_LINE.match(line)]
    if not option_lines or option_lines[-1] == 0:
        return None
    i = option_lines[-1]
    previous = lines[i - 1]
    if _OPTION_LINE.match(previous) or not previous.rstrip("*_ ").endswith(":"):
        return None
    match = _OPTION_LINE.match(lines[i])
    return match.group(1) if match else None


def extract_choice(prediction: str) -> str | None:
    marked = _MARKED.findall(prediction)
    if marked:
        letter: str = marked[-1]
        return letter
    boxed = extract_boxed(prediction)
    if boxed is not None and (match := _BOXED_LETTER.match(boxed)):
        return match.group(1)
    concluding = _concluding_option_line(prediction)
    if concluding is not None:
        return concluding
    text = prediction.strip()
    for pattern in (_BARE, _LEADING):
        match = pattern.match(text)
        if match:
            return match.group(1)
    return None


def grade_choice(prediction: str, reference: str) -> bool:
    expected = reference.strip().strip("()").upper()
    if len(expected) != 1 or not "A" <= expected <= "J":
        raise ValueError(f"reference is not an option letter: {reference!r}")
    return extract_choice(prediction) == expected
