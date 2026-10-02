import pytest

from switchboard.labeling.graders.bbh import grade_bbh


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        # Option targets.
        ("Let's think step by step... So the answer is (C).", "(C)", True),
        ("The answer is C", "(C)", True),
        ("The answer is (B) Sam", "(B)", True),
        ("The answer is (A).", "(C)", False),
        # Boolean-ish targets.
        ("not ( True ) and ( True ) is False. The answer is False.", "False", True),
        ("The answer is: Yes, the argument holds.", "Yes", True),
        ("The answer is valid", "invalid", False),
        ("The answer is invalid.", "invalid", True),
        ("The answer is No", "Yes", False),
        # Integer targets.
        ("((-1 + 2) * 3) = 3. The answer is 3.", "3", True),
        ("The answer is -12", "-12", True),
        ("The answer is 7", "6", False),
        # Free-text targets.
        ("The answer is: apple banana cherry", "apple banana cherry", True),
        ("The answer is apple cherry banana", "apple banana cherry", False),
        # Bracket sequences, compared without whitespace.
        ("The answer is ) ] }", ") ] }", True),
        ("The answer is )]}", ") ] }", True),
        ("The answer is ] ) }", ") ] }", False),
        # No marker: the last line is the answer.
        ("Reasoning here.\n(D)", "(D)", True),
        # Empty and refusal.
        ("", "(A)", False),
        ("I'm sorry, I can't help with that.", "Yes", False),
    ],
)
def test_fixtures(prediction: str, reference: str, expected: bool) -> None:
    assert grade_bbh(prediction, reference) is expected


# --- Answer forms found in the R1 label review (round 1) ----------------------------------------


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        # web_of_lies-248 (review round 1, #29): a statement instead of Yes/No.
        (
            "Therefore, Jim does not tell the truth.\n\nThe answer is Jim does not tell the truth.",
            "No",
            True,
        ),
        ("The answer is Jim does not tell the truth.", "Yes", False),
        ("The answer is Ryan tells the truth.", "Yes", True),
        ("The answer is Ryan tells the truth.", "No", False),
        ("Final line: The answer is Vina lies.", "No", True),
        ("Therefore, the answer is that Fletcher tells the truth.", "Yes", True),
        # Bold "Final Answer" markers.
        ("Reasoning...\n\n**Final Answer**: No.", "No", True),
        ("Reasoning...\n\n**Final Answer**: No.", "Yes", False),
        # The answer word is not the first word.
        ("Reasoning...\n\n**Final Answer:** The argument is valid.", "valid", True),
        ("Reasoning...\n\n**Final Answer:** The argument is valid.", "invalid", False),
        ("The answer is: the argument is not valid.", "invalid", True),
        ("Therefore, the result of not True or True and False and False is False.", "False", True),
        ("Therefore, the result of not True or True and False and False is False.", "True", False),
        ("Simplifying step by step.\n(False).", "False", True),
        ("Yes.)", "Yes", True),
        # The first word still wins when it is an answer word.
        ("The answer is No, even though one might first say yes.", "No", True),
        # A sentence with no answer word stays wrong: the model did not answer the question asked.
        ("The answer is Jen did not intend to kill the puppies.", "No", False),
    ],
)
def test_review_round1_answer_forms(prediction: str, reference: str, expected: bool) -> None:
    assert grade_bbh(prediction, reference) is expected


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        # word_sorting-152 (review round 2, #20): correct order, commas and capitals.
        ("The answer is Buckley, Frisian, IX, Livre, Panoramic, Substitution.", True),
        ("The answer is buckley frisian ix livre panoramic substitution", True),
        # Order still matters, and no word may be missing or extra.
        ("The answer is Frisian, Buckley, IX, Livre, Panoramic, Substitution.", False),
        ("The answer is Buckley, Frisian, IX, Livre, Panoramic.", False),
        ("The answer is Buckley, Frisian, IX, Livre, Panoramic, Substitution, Zebra.", False),
    ],
)
def test_review_round2_word_lists(prediction: str, expected: bool) -> None:
    reference = "buckley frisian ix livre panoramic substitution"
    assert grade_bbh(prediction, reference) is expected
