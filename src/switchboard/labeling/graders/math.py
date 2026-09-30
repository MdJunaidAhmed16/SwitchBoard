"""MATH: extract the final answer, normalise LaTeX, compare as strings then as numbers.

Normalisation follows the reference implementation released with the MATH benchmark
(Hendrycks et al., ``strip_string``), plus a numeric-equivalence fallback so ``\\frac{1}{2}``
and ``0.5`` agree.
"""

from __future__ import annotations

import re
from fractions import Fraction

from switchboard.labeling.graders.common import extract_boxed, last_answer_clause

_TEXT_WRAPPER = re.compile(r"\\(?:text|textbf|mathrm|mbox)\{([^{}]*)\}")
_UNITS = re.compile(
    r"\\text\{\s*(?:cm|m|km|inches|inch|feet|foot|ft|units?|degrees?|sq|square|cents?|dollars?)"
    r"[^{}]*\}"
)


def extract_math_answer(prediction: str) -> str | None:
    boxed = extract_boxed(prediction)
    if boxed is not None:
        return boxed
    clause = last_answer_clause(prediction)
    if clause is None:
        return None
    return clause.rstrip(".").strip() or None


def _fix_fracs(s: str) -> str:
    """``\\frac12`` -> ``\\frac{1}{2}``, ``\\frac1{2}`` -> ``\\frac{1}{2}``."""
    parts = s.split("\\frac")
    out = parts[0]
    for part in parts[1:]:
        out += "\\frac"
        if part.startswith("{") or len(part) < 2:
            out += part
            continue
        a, b, rest = part[0], part[1], part[2:]
        if b == "{":
            out += "{" + a + "}{" + rest
        else:
            out += "{" + a + "}{" + b + "}" + rest
    return out


def _fix_a_slash_b(s: str) -> str:
    if s.count("/") != 1:
        return s
    a, b = s.split("/")
    if a.lstrip("-").isdigit() and b.isdigit():
        return f"\\frac{{{a}}}{{{b}}}"
    return s


def _fix_sqrt(s: str) -> str:
    """``\\sqrt3`` -> ``\\sqrt{3}``."""
    return re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)


def normalize_math(answer: str) -> str:
    s = answer.strip()
    s = s.replace("\n", "").replace("\\!", "")
    s = s.replace("\\\\", "\\")
    s = s.replace("tfrac", "frac").replace("dfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = s.replace("\\$", "").replace("$", "")
    s = _UNITS.sub("", s)
    s = _TEXT_WRAPPER.sub(r"\1", s)
    s = s.replace("\\%", "").replace("%", "")
    s = s.replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace(" .", " 0.").replace("{.", "{0.")
    if s.startswith("."):
        s = "0" + s
    # "x = 5" -> "5" when the left side is a short variable name.
    if s.count("=") == 1 and len(s.split("=")[0].strip()) <= 2:
        s = s.split("=")[1]
    s = _fix_sqrt(s)
    s = s.replace(" ", "")
    s = _fix_fracs(s)
    if s == "0.5":
        s = "\\frac{1}{2}"
    s = _fix_a_slash_b(s)
    return s.rstrip(".")


_SIMPLE_FRAC = re.compile(r"^(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}$")


def _as_number(s: str) -> Fraction | None:
    """Parse a normalised answer that is a plain integer, decimal or simple fraction."""
    s = s.replace(",", "")
    match = _SIMPLE_FRAC.match(s)
    if match:
        sign, num, den = match.groups()
        if int(den) == 0:
            return None
        value = Fraction(int(num), int(den))
        return -value if sign else value
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def grade_math(prediction: str, reference: str) -> bool:
    got = extract_math_answer(prediction)
    if got is None:
        return False
    a, b = normalize_math(got), normalize_math(reference)
    if a == b:
        return True
    na, nb = _as_number(a), _as_number(b)
    return na is not None and nb is not None and na == nb
