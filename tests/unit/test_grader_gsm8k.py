import pytest

from switchboard.labeling.graders.numeric import extract_final_number, grade_gsm8k


@pytest.mark.parametrize("prediction", ["42", "42.0", " 42 ", "The answer is 42"])
def test_normalisation_variants_agree(prediction: str) -> None:
    # 13-testing-and-reports: these four must not disagree.
    assert grade_gsm8k(prediction, "42")


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        # Correct, surrounded by working and prose.
        ("She has 5 + 3 = 8 apples.\nThe answer is: 8", "8", True),
        ("Step 1: 12*3=36\nStep 2: 36-6=30\nThe answer is: 30 dollars.", "30", True),
        # Correct number, different surface format.
        ("The answer is: $1,250.00", "1250", True),
        ("The answer is **18**.", "18", True),
        ("#### 72", "Natalia sold 48/2 = 24 clips.\n#### 72", True),
        ("Final answer: \\boxed{15}", "15", True),
        ("The answer is -7", "-7", True),
        # Last marker wins over earlier intermediate values.
        ("The answer is 10? No, recheck. The answer is 12.", "12", True),
        # No marker: fall back to the last number.
        ("After paying 3 dollars she keeps 17 dollars.", "17", True),
        # Wrong answers.
        ("The answer is 41", "42", False),
        ("The answer is 4.2", "42", False),
        ("The answer is 420", "42", False),
        # Marker present but the last number elsewhere is not what counts.
        ("The answer is 9.\nCheck: 9 * 2 = 18", "18", False),
        # Empty generation and refusal.
        ("", "42", False),
        ("I'm sorry, but I can't help with that request.", "42", False),
    ],
)
def test_fixtures(prediction: str, reference: str, expected: bool) -> None:
    assert grade_gsm8k(prediction, reference) is expected


def test_reference_without_number_is_a_loud_error() -> None:
    with pytest.raises(ValueError, match="no number"):
        grade_gsm8k("42", "#### ")


def test_thousands_separator_is_not_two_numbers() -> None:
    assert extract_final_number("It costs 1,000 dollars") == 1000
