import pytest

from switchboard.labeling.graders.math import extract_math_answer, grade_math, normalize_math


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("So the result is $\\boxed{5}$.", "5", True),
        ("Therefore $x = \\boxed{\\frac{1}{2}}$.", "\\frac{1}{2}", True),
        # Equivalent forms of the same value.
        ("\\boxed{\\dfrac{1}{2}}", "\\frac{1}{2}", True),
        ("\\boxed{0.5}", "\\frac{1}{2}", True),
        ("\\boxed{1/2}", "\\frac12", True),
        ("\\boxed{\\frac{6}{4}}", "\\frac{3}{2}", True),
        ("\\boxed{x = 3}", "3", True),
        ("\\boxed{90^\\circ}", "90", True),
        ("\\boxed{\\text{(C)}}", "\\text{(C)}", True),
        ("\\boxed{\\left( 3, -1 \\right)}", "(3,-1)", True),
        ("\\boxed{10\\%}", "10", True),
        ("\\boxed{2\\sqrt{3}}", "2\\sqrt3", True),
        # Nested braces inside the box.
        ("\\boxed{\\frac{\\sqrt{3}}{2}}", "\\frac{\\sqrt{3}}{2}", True),
        # Last box wins.
        ("First guess \\boxed{4}, corrected: \\boxed{6}", "6", True),
        # No box, explicit marker.
        ("The answer is 12.", "12", True),
        # Wrong.
        ("\\boxed{\\frac{1}{3}}", "\\frac{1}{2}", False),
        ("\\boxed{(3,1)}", "(3,-1)", False),
        ("\\boxed{-5}", "5", False),
        # No extractable answer: a bare number in prose is not accepted for MATH.
        ("I computed it and got 5", "5", False),
        ("", "5", False),
        ("I cannot solve this problem.", "5", False),
    ],
)
def test_fixtures(prediction: str, reference: str, expected: bool) -> None:
    assert grade_math(prediction, reference) is expected


def test_unterminated_box_yields_nothing() -> None:
    assert extract_math_answer("\\boxed{\\frac{1}{2}") is None


def test_normalisation_is_idempotent() -> None:
    once = normalize_math("\\dfrac{1}{2}")
    assert normalize_math(once) == once
