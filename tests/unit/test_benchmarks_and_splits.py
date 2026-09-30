import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from switchboard.config import Settings
from switchboard.labeling.benchmarks import (
    REGISTRY,
    Item,
    build_arc,
    build_gsm8k,
    build_mbpp,
    build_mmlu,
    hash_sample,
)
from switchboard.labeling.splits import load_splits

INSTRUCTION_MARKERS = ("The answer is", "\\boxed", "```python", "step by step")


def _item(i: int) -> Item:
    return Item("b", f"id-{i:04d}", "t", "t", "r")


def test_hash_sample_is_deterministic_and_seed_dependent() -> None:
    items = [_item(i) for i in range(100)]
    a = hash_sample(items, 10, seed=0)
    assert a == hash_sample(list(reversed(items)), 10, seed=0)
    assert a != hash_sample(items, 10, seed=1)
    assert len(a) == 10


def test_hash_sample_keeps_everything_under_the_limit() -> None:
    items = [_item(i) for i in range(5)]
    assert hash_sample(items, 10, seed=0) == items
    assert hash_sample(items, None, seed=0) == items


def test_gsm8k_reference_is_the_final_number() -> None:
    df = pd.DataFrame(
        {
            "question": ["Q?"],
            "answer": ["work 1,000 + 200\n#### 1,200"],
            "_split": ["test"],
            "_row": [3],
        }
    )
    (item,) = build_gsm8k(df)
    assert item.reference == "1200"
    assert item.item_id == "test-00003"


def test_router_text_carries_no_format_instruction() -> None:
    df = pd.DataFrame(
        {
            "question": ["Which is a mammal?"],
            "choices": [np.array(["Cat", "Trout", "Frog", "Hawk"])],
            "answer": [0],
            "_row": [0],
        }
    )
    (item,) = build_mmlu(df)
    for marker in INSTRUCTION_MARKERS:
        assert marker not in item.router_text
    assert item.user_prompt.startswith(item.router_text)
    assert "A. Cat" in item.router_text
    assert item.reference == "A"


def test_arc_numeric_labels_are_relettered_by_position() -> None:
    df = pd.DataFrame(
        {
            "id": ["x1"],
            "question": ["Q?"],
            "choices": [{"text": np.array(["a", "b", "c"]), "label": np.array(["1", "2", "3"])}],
            "answerKey": ["3"],
        }
    )
    (item,) = build_arc(df)
    assert item.reference == "C"
    assert "C. c" in item.router_text


def test_mbpp_reference_holds_tests_and_setup() -> None:
    df = pd.DataFrame(
        {
            "task_id": [7],
            "text": ["Write f."],
            "test_list": [np.array(["assert f() == 1"])],
            "test_setup_code": ["X = 1"],
        }
    )
    (item,) = build_mbpp(df)
    assert json.loads(item.reference) == {
        "test_setup_code": "X = 1",
        "test_list": ["assert f() == 1"],
    }
    assert item.item_id == "mbpp-0007"


def test_every_source_is_pinned_to_a_commit() -> None:
    for bench in REGISTRY.values():
        assert len(bench.revision) == 40, bench.name
        int(bench.revision, 16)


# --- Splits ------------------------------------------------------------------------------------


def test_committed_splits_are_disjoint_and_complete() -> None:
    roles = load_splits(Settings().splits_path, known=set(REGISTRY))
    assert set(roles) == set(REGISTRY)
    assert {r for r in roles.values()} == {"train", "val", "test"}


def test_overlapping_split_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("train: [gsm8k]\nval: [arc_challenge]\ntest: [gsm8k]\n")
    with pytest.raises(ValueError, match="both"):
        load_splits(path)


def test_unknown_benchmark_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("train: [gsm8k]\nval: [arc_challenge]\ntest: [nope]\n")
    with pytest.raises(ValueError, match="unknown"):
        load_splits(path, known=set(REGISTRY))
