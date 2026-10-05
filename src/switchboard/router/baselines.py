"""The two baselines, behind the same interface as the learned routers.

**Heuristic** — what a sceptic says you could build in an afternoon without ML: prompt length plus
a small keyword list, mapped to a score by logistic regression on exactly those two features
(05-router-model). If it matches the learned router, the learned router is unnecessary.

**Random** — a uniform score that ignores the prompt. At any threshold it escalates a fraction of
traffic at random, which is the "random routing at the same escalation rate" reference. If a
learned router does no better, it has learned nothing.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from switchboard.router.interface import Scores

# Fixed before any router was evaluated, and never tuned against results. Words that signal a
# demanding task: proofs, derivations, implementation work, exhaustive or multi-step reasoning.
KEYWORDS: tuple[str, ...] = (
    "prove",
    "derive",
    "implement",
    "show that",
    "find all",
    "how many",
    "probability",
    "polynomial",
    "integer",
    "algorithm",
    "function",
    "step",
)
_WORD = re.compile(r"\S+")


def heuristic_features(texts: Sequence[str]) -> np.ndarray:
    """Two columns: log(1 + whitespace-token count), number of keywords present."""
    rows = []
    for text in texts:
        lowered = text.lower()
        n_words = len(_WORD.findall(text))
        n_keywords = sum(1 for k in KEYWORDS if re.search(rf"\b{re.escape(k)}\b", lowered))
        rows.append((math.log1p(n_words), float(n_keywords)))
    return np.asarray(rows, dtype=np.float64)


class HeuristicRouter:
    name = "heuristic"

    def __init__(self) -> None:
        self._model = make_pipeline(StandardScaler(), LogisticRegression())

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        # No hyperparameters, so validation is not used.
        self._model.fit(heuristic_features(list(train["router_text"])), train["label"].astype(int))

    def predict_proba(self, texts: Sequence[str]) -> Scores:
        proba: Scores = self._model.predict_proba(heuristic_features(texts))[:, 1]
        return proba.astype(np.float64)


class RandomRouter:
    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        return None

    def predict_proba(self, texts: Sequence[str]) -> Scores:
        # A hash of (seed, text) rather than a stateful RNG: the same prompt gets the same score
        # whatever order it is scored in, so results are reproducible.
        out = np.empty(len(texts), dtype=np.float64)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(f"{self.seed}:{text}".encode()).digest()
            out[i] = int.from_bytes(digest[:8], "big") / 2**64
        return out
