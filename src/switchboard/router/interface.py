"""The one interface every router implements: heuristic, random, v0 and v1 alike.

The sweep and the evaluation treat all routers through this interface only, so none can be
favoured by accident (02-architecture, module boundaries).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Scores = NDArray[np.float64]


@runtime_checkable
class Router(Protocol):
    name: str

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        """Learn from ``train`` (columns ``router_text``, ``label``); tune on ``val`` only."""
        ...

    def predict_proba(self, texts: Sequence[str]) -> Scores:
        """P(the local model answers correctly), one value in [0, 1] per text."""
        ...
