"""BIG-Bench Hard: exact match on the final answer.

Targets come in four shapes: an option like ``(C)``, a boolean-ish word (``True``, ``No``,
``invalid``), an integer, or free text (sorted words, bracket sequences, names). Each is compared
in the way that shape needs, and nothing looser.
"""

from __future__ import annotations

import re

from switchboard.labeling.graders.common import last_answer_clause, parse_number

_OPTION_TARGET = re.compile(r"^\(([A-R])\)$")
_OPTION_IN_TEXT = re.compile(r"\(([A-R])\)|^([A-R])(?![A-Za-z])")
_WORD_TARGETS = {"true", "false", "yes", "no", "valid", "invalid"}
_STRIP = " \t\"'`*.,:;"


def _final_clause(prediction: str) -> str | None:
    clause = last_answer_clause(prediction)
    if clause is not None:
        return clause
    lines = [line for line in prediction.strip().splitlines() if line.strip()]
    return lines[-1] if lines else None


def _norm(text: str) -> str:
    return " ".join(text.strip(_STRIP).lower().split())


def grade_bbh(prediction: str, reference: str) -> bool:
    clause = _final_clause(prediction)
    if clause is None:
        return False
    target = reference.strip()

    option = _OPTION_TARGET.match(target)
    if option:
        found = _OPTION_IN_TEXT.search(clause.strip(_STRIP))
        return found is not None and (found.group(1) or found.group(2)) == option.group(1)

    if _norm(target) in _WORD_TARGETS:
        words = _norm(clause).split()
        return bool(words) and words[0].strip(_STRIP) == _norm(target)

    if re.fullmatch(r"-?\d+", target):
        got = parse_number(clause)
        return got is not None and got == int(target)

    got_text, want_text = _norm(clause), _norm(target)
    if got_text == want_text:
        return True
    # Bracket-sequence targets (dyck_languages) are compared without whitespace.
    if not re.search(r"[A-Za-z0-9]", target):
        return "".join(got_text.split()) == "".join(want_text.split())
    return False
