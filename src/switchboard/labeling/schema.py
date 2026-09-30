"""Schema of ``data/labels/<model>.parquet``. Checked on every write."""

from __future__ import annotations

import pandas as pd

# column -> pandas dtype kind ("O" object/string, "b" bool, "i" int, "f" float)
LABEL_COLUMNS: dict[str, str] = {
    "benchmark": "O",
    "item_id": "O",
    "prompt_hash": "O",
    "router_text": "O",
    "user_prompt": "O",
    "reference": "O",
    "generation": "O",
    "label": "b",
    "prompt_tokens": "i",
    "output_tokens": "i",
    "latency_ms": "f",
    "finish_reason": "O",
    "model_id": "O",
    "model_revision": "O",
    "decode_params_hash": "O",
}


def _kind(series: pd.Series) -> str:
    kind = series.dtype.kind
    # pandas 3 string dtype reports kind "O" or "T"; treat both as text.
    return "O" if kind in ("O", "T", "U") else str(kind)


def validate_labels(df: pd.DataFrame) -> None:
    missing = set(LABEL_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"labels missing columns: {sorted(missing)}")
    nulls = [c for c in LABEL_COLUMNS if df[c].isna().any()]
    if nulls:
        raise ValueError(f"labels have nulls in: {nulls}")
    wrong = {c: _kind(df[c]) for c in LABEL_COLUMNS if _kind(df[c]) != LABEL_COLUMNS[c]}
    if wrong:
        raise ValueError(f"labels have wrong dtypes: {wrong}")
    if df.duplicated(["benchmark", "item_id"]).any():
        raise ValueError("labels have duplicate (benchmark, item_id) rows")
