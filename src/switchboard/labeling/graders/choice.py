"""Multiple choice (MMLU, ARC): extract a single option letter."""

from __future__ import annotations

import re

# The letter must be uppercase: "the answer is a function" must not read as option A.
_MARKED = re.compile(
    r"(?i:answer\s*(?:is)?\s*[:\-]?\s*(?:option\s*)?)[\*\s]*\(?([A-J])\)?(?![A-Za-z])"
)
_BARE = re.compile(r"^[\*\s]*\(?([A-J])\)?[\.\:\)]?[\*\s]*$")
_LEADING = re.compile(r"^[\*\s]*\(?([A-J])[\.\:\)]\s")


def extract_choice(prediction: str) -> str | None:
    marked = _MARKED.findall(prediction)
    if marked:
        letter: str = marked[-1]
        return letter
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
