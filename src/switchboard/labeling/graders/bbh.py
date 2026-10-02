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
_WORD_FAMILIES = ({"true", "false"}, {"yes", "no"}, {"valid", "invalid"})
_WORD_TARGETS = set().union(*_WORD_FAMILIES)
_STRIP = " \t\"'`*.,:;"
_WORD = re.compile(r"\b(true|false|yes|no|valid|invalid)\b", re.IGNORECASE)
_NOT_VALID = re.compile(r"\bnot\s+valid\b", re.IGNORECASE)
# Truth-teller questions ("Does Jim tell the truth?") are often answered as a statement.
_SAYS_LIES = re.compile(
    r"\b(?:do(?:es)?\s*n[o']?t|not)\s+tell(?:s|ing)?\s+the\s+truth\b"
    r"|\b(?:lies|is\s+lying|is\s+a\s+liar)\b",
    re.IGNORECASE,
)
_SAYS_TRUTH = re.compile(r"\b(?:tells|is\s+telling)\s+the\s+truth\b|\bis\s+truthful\b", re.I)


def _final_clause(prediction: str) -> str | None:
    clause = last_answer_clause(prediction)
    if clause is not None:
        return clause
    lines = [line for line in prediction.strip().splitlines() if line.strip()]
    return lines[-1] if lines else None


def _norm(text: str) -> str:
    return " ".join(text.strip(_STRIP).lower().split())


def _loose(text: str) -> str:
    return " ".join(re.sub(r"[,.;:\"'`*()\[\]]", " ", text.lower()).split())


def _word_answer(clause: str, target: str) -> str | None:
    """The yes/no, true/false or valid/invalid answer given in ``clause``, if any.

    The first word wins when it is an answer word ("No, because ..."). Otherwise the last answer
    word of the target's family ("... is False.", "The argument is valid."). For yes/no, a clause
    that only states someone lies or tells the truth is read as No or Yes.
    """
    family = next(f for f in _WORD_FAMILIES if target in f)
    words = _norm(clause).split()
    first = words[0].strip(_STRIP + "()") if words else ""
    if first in family:
        return first
    text = _NOT_VALID.sub("invalid", clause)
    hits = [m.group(1).lower() for m in _WORD.finditer(text) if m.group(1).lower() in family]
    if hits:
        return hits[-1]
    if family == {"yes", "no"}:
        if _SAYS_LIES.search(clause):
            return "no"
        if _SAYS_TRUTH.search(clause):
            return "yes"
    return None


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
        return _word_answer(clause, _norm(target)) == _norm(target)

    if re.fullmatch(r"-?\d+", target):
        got = parse_number(clause)
        return got is not None and got == int(target)

    got_text, want_text = _norm(clause), _norm(target)
    if got_text == want_text:
        return True
    # Word-list targets (word_sorting) ignore separators and case, so a correctly ordered
    # "Buckley, Frisian, IX." matches "buckley frisian ix". Word order still has to match.
    if re.search(r"[A-Za-z]", target):
        return _loose(clause) == _loose(target)
    # Bracket-sequence targets (dyck_languages) are compared without whitespace.
    if not re.search(r"[A-Za-z0-9]", target):
        return "".join(got_text.split()) == "".join(want_text.split())
    return False
