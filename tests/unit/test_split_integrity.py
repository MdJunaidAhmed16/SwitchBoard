"""The most important test in the repository (13-testing-and-reports, Phase 2).

No prompt may appear in more than one of train / val / test. It runs against the committed labels
file and is not marked or skippable: if the labels file is missing, the test fails.
"""

import pandas as pd
import pytest

from switchboard.config import Settings
from switchboard.router.data import check_split_integrity, load_router_data, row_split, text_key


def test_committed_labels_have_no_cross_split_prompts() -> None:
    data = load_router_data(Settings())  # runs check_split_integrity on the real labels
    keys = {role: set(frame["router_text"].map(text_key)) for role, frame in
            (("train", data.train), ("val", data.val), ("test", data.test))}  # fmt: skip
    assert not keys["train"] & keys["test"]
    assert not keys["train"] & keys["val"]
    assert not keys["val"] & keys["test"]


def test_roles_follow_splits_yaml() -> None:
    data = load_router_data(Settings())
    assert set(data.train["benchmark"]) == {"gsm8k", "mmlu", "mbpp"}
    assert set(data.val["benchmark"]) == {"arc_challenge"}
    assert set(data.test["benchmark"]) == {"math", "humaneval", "bbh"}


def _frame(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["benchmark", "item_id", "router_text"]).assign(
        role=lambda d: d["benchmark"].map({"a": "train", "b": "test"})
    )


def test_leak_is_detected() -> None:
    df = _frame([("a", "1", "What is 2+2?"), ("b", "2", "What is 2+2?")])
    with pytest.raises(ValueError, match="more than one split"):
        check_split_integrity(df)


def test_leak_is_detected_despite_case_and_spacing() -> None:
    df = _frame([("a", "1", "What  is 2+2?"), ("b", "2", "what is 2+2? ")])
    with pytest.raises(ValueError, match="more than one split"):
        check_split_integrity(df)


def test_duplicates_within_one_split_are_allowed() -> None:
    check_split_integrity(_frame([("a", "1", "Same text"), ("a", "2", "Same text")]))


def test_row_split_is_deterministic_and_disjoint() -> None:
    df = pd.DataFrame({"benchmark": ["x"] * 100, "item_id": [str(i) for i in range(100)]})
    train, test = row_split(df, test_fraction=0.2, seed=0)
    again, _ = row_split(df, test_fraction=0.2, seed=0)
    assert len(test) == 20
    assert len(train) == 80
    assert not set(train["item_id"]) & set(test["item_id"])
    assert train["item_id"].tolist() == again["item_id"].tolist()
