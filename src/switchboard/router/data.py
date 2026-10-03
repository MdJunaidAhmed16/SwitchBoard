"""Labels → train / val / test frames for the router, split by dataset and checked for leakage."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pandas as pd

from switchboard.config import Settings
from switchboard.labeling.benchmarks import REGISTRY
from switchboard.labeling.generate import labels_path
from switchboard.labeling.splits import load_splits


@dataclass(frozen=True)
class RouterData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def all(self) -> pd.DataFrame:
        return pd.concat([self.train, self.val, self.test], ignore_index=True)


def text_key(text: str) -> str:
    """Identity of a prompt for leakage checks: case- and whitespace-insensitive."""
    return hashlib.sha256(re.sub(r"\s+", " ", text.lower()).strip().encode("utf-8")).hexdigest()


def check_split_integrity(df: pd.DataFrame) -> None:
    """Fail if any prompt text appears under more than one role.

    This is the guard behind the dataset-level split (04-datasets): a prompt seen in training must
    never be scored as a held-out example.
    """
    keys = df["router_text"].map(text_key)
    roles_per_key = df.groupby(keys)["role"].nunique()
    leaked = roles_per_key[roles_per_key > 1]
    if len(leaked):
        examples = df[keys.isin(leaked.index[:5])][["benchmark", "item_id", "role"]]
        raise ValueError(f"{len(leaked)} prompt(s) appear in more than one split, e.g.\n{examples}")


def load_router_data(settings: Settings) -> RouterData:
    df = pd.read_parquet(labels_path(settings))
    roles = load_splits(settings.splits_path, known=set(REGISTRY))
    df = df.assign(role=df["benchmark"].map(roles))
    if df["role"].isna().any():
        missing = sorted(df.loc[df["role"].isna(), "benchmark"].unique())
        raise ValueError(f"benchmarks with no split role: {missing}")
    check_split_integrity(df)
    by_role = {r: df[df["role"] == r].reset_index(drop=True) for r in ("train", "val", "test")}
    return RouterData(train=by_role["train"], val=by_role["val"], test=by_role["test"])


def row_split(
    df: pd.DataFrame, test_fraction: float = 0.2, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A random row-level split of all rows, for the *optimistic* comparison only.

    Rows from every benchmark land on both sides, so a router can learn which benchmark a prompt
    came from. It is reported next to the dataset-level number, labelled optimistic, never alone.
    """
    order = (df["benchmark"] + ":" + df["item_id"]).map(
        lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest()
    )
    ranked = df.assign(_order=order).sort_values("_order").drop(columns="_order")
    n_test = round(len(ranked) * test_fraction)
    test = ranked.iloc[:n_test].reset_index(drop=True)
    train = ranked.iloc[n_test:].reset_index(drop=True)
    return train, test
