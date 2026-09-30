import pytest

from switchboard.labeling.graders.choice import extract_choice, grade_choice


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("The answer is (B)", "B", True),
        ("The answer is B.", "B", True),
        ("Answer: C", "C", True),
        ("The answer is **D**", "D", True),
        ("Photosynthesis happens in chloroplasts.\nThe answer is (A)", "A", True),
        ("B", "B", True),
        ("(C)", "C", True),
        ("D. Mitochondria, because it produces ATP.", "D", True),
        ("The answer is (E)", "E", True),
        # Last marker wins.
        ("I first thought the answer is (A), but the answer is (C).", "C", True),
        # Wrong letter.
        ("The answer is (A)", "B", False),
        # Lowercase article must not read as option A.
        ("The answer is a mitochondrion.", "A", False),
        # Pronoun "I" must not read as an option.
        ("I think it is related to energy.", "I", False),
        # Empty and refusal.
        ("", "A", False),
        ("I'm sorry, I cannot answer that question.", "A", False),
    ],
)
def test_fixtures(prediction: str, reference: str, expected: bool) -> None:
    assert grade_choice(prediction, reference) is expected


def test_reference_must_be_a_letter() -> None:
    with pytest.raises(ValueError, match="option letter"):
        grade_choice("A", "Paris")


def test_parenthesised_reference_is_accepted() -> None:
    assert grade_choice("The answer is (B)", "(B)")


def test_no_letter_extracted_from_prose() -> None:
    assert extract_choice("Energy is stored in bonds.") is None
