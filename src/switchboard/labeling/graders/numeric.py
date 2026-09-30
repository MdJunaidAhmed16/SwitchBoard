"""GSM8K: exact numeric match after normalisation."""

from __future__ import annotations

from decimal import Decimal

from switchboard.labeling.graders.common import (
    extract_boxed,
    last_answer_clause,
    last_number,
    parse_number,
)


def extract_final_number(prediction: str) -> Decimal | None:
    """The model's final numeric answer, preferring explicit markers over the last number seen."""
    clause = last_answer_clause(prediction)
    if clause is not None and (value := parse_number(clause)) is not None:
        return value
    if "####" in prediction:
        value = parse_number(prediction.rsplit("####", 1)[1])
        if value is not None:
            return value
    boxed = extract_boxed(prediction)
    if boxed is not None and (value := parse_number(boxed)) is not None:
        return value
    return last_number(prediction)


def grade_gsm8k(prediction: str, reference: str) -> bool:
    expected = parse_number(reference.rsplit("####", 1)[-1])
    if expected is None:
        raise ValueError(f"reference has no number: {reference!r}")
    got = extract_final_number(prediction)
    # Decimal equality: "42" == "42.0" == "42.00".
    return got is not None and got == expected
