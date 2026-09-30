"""Answer-extraction helpers shared by the graders. Pure functions, no I/O."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

# "The answer is: X", "Final answer: X", "answer is X". The clause runs to end of line.
# A dash counts as a separator only when it is not a minus sign ("answer is -7").
_ANSWER_IS = re.compile(r"answer\s*(?:is|:)\s*(?::|-(?![\d.]))?\s*", re.IGNORECASE)

# Thousands-separated numbers first, so "1,000" is one number and "1, 2" is two.
_NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?|-?\.\d+")

_MARKUP = str.maketrans("", "", "*`$")


def last_answer_clause(text: str) -> str | None:
    """Text after the last "answer is"/"answer:" marker, on the same line, markup stripped."""
    # Anchor on the last marker; a greedy capture would swallow a later marker on the same line.
    markers = list(_ANSWER_IS.finditer(text))
    if not markers:
        return None
    last = markers[-1]
    clause = text[last.end() :].split("\n", 1)[0].translate(_MARKUP).strip()
    return clause or None


def parse_number(text: str) -> Decimal | None:
    """First number in ``text`` as a Decimal, ignoring thousands separators."""
    match = _NUMBER.search(text)
    if match is None:
        return None
    try:
        return Decimal(match.group(0).replace(",", ""))
    except InvalidOperation:
        return None


def last_number(text: str) -> Decimal | None:
    matches = _NUMBER.findall(text)
    if not matches:
        return None
    try:
        return Decimal(matches[-1].replace(",", ""))
    except InvalidOperation:
        return None


def extract_boxed(text: str) -> str | None:
    """Contents of the last ``\\boxed{...}`` (or ``\\fbox{...}``), with balanced braces."""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    i = text.find("{", start)
    if i < 0:
        # "\boxed 5" form: take the next whitespace-delimited token.
        rest = text[start:].split(None, 2)
        return rest[1].rstrip("$.") if len(rest) > 1 else None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j]
    return None
