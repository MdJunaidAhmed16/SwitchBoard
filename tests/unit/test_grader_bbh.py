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
