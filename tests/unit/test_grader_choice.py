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


# --- Answer forms found in the R1 label review (round 1) ----------------------------------------

ENUMERATION = (
    "Let's analyze each option:\n\n"
    "A. It converts sunlight to energy: incorrect.\n"
    "B. It controls all functions: incorrect.\n"
    "C. It transports water: incorrect.\n"
    "D. It breaks down sugar: correct, this is cellular respiration."
)


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        # mmlu/test-10716 (review round 1, #21): a concluding option line after a colon.
        (
            "### Conclusion:\nGiven the context, the most appropriate action would be:\n\n"
            "**B. No, unless the court finds ... prejudicial effect.**\n\n"
            "This approach ensures that the court considers all relevant factors.",
            "B",
            True,
        ),
        (
            "To conserve natural resources, we should choose:\n\nA. repair your TV.\n\nWhy...",
            "A",
            True,
        ),
        # Boxed letters.
        ("Therefore, the correct answer is:\n\n\\boxed{B}", "B", True),
        ("So, the correct choice is:\n\n\\[\n\\boxed{\\text{B}}\n\\]", "B", True),
        ("The answer is \\(\\boxed{D}\\).", "D", True),
        ("\\boxed{12}", "A", False),
        # Other explicit phrasings.
        ("After analyzing all the options, the correct choice is C.", "C", True),
        ("The best option is D because it is renewable.", "D", True),
        # An enumeration with no conclusion must not yield its last option.
        (ENUMERATION, "D", False),
        (ENUMERATION + "\n\nConsidering all of this, the answer is:\n\n**D. sugar**", "D", True),
        # "Option A: ..." analysis lines are not answer markers.
        ("Option A: incorrect.\nOption B: incorrect.\nOption C: maybe.", "C", False),
        # A concluding line that is not preceded by a colon is not accepted.
        ("Thinking about it more.\nB. Something here.\nMore text.", "B", False),
    ],
)
def test_review_round1_answer_forms(prediction: str, reference: str, expected: bool) -> None:
    assert grade_choice(prediction, reference) is expected
