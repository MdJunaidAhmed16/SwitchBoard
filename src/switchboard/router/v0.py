"""Router v0: frozen sentence-encoder embeddings + logistic regression.

Cheap and fast; establishes whether prompt embeddings carry any difficulty signal at all. The
regularisation strength is the only hyperparameter and is chosen on the validation benchmark,
never on test.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from switchboard.router.encoder import Embedder
from switchboard.router.interface import Scores
from switchboard.router.metrics import auroc

C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)


class V0Router:
    name = "v0"

    def __init__(self, embedder: Embedder, c_grid: Sequence[float] = C_GRID, seed: int = 0) -> None:
        self.embedder = embedder
        self.c_grid = tuple(c_grid)
        self.seed = seed
        self.chosen_c: float | None = None
        self.val_auroc_by_c: dict[float, float] = {}
        self._model: LogisticRegression | None = None

    def _new_model(self, c: float) -> LogisticRegression:
        return LogisticRegression(C=c, max_iter=2000, random_state=self.seed)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        x_train = self.embedder.embed(list(train["router_text"]))
        y_train = train["label"].astype(int).to_numpy()
        x_val = self.embedder.embed(list(val["router_text"]))
        y_val = val["label"].astype(int).to_numpy()
        best: tuple[float, float, LogisticRegression] | None = None
        for c in self.c_grid:
            model = self._new_model(c).fit(x_train, y_train)
            score = auroc(y_val, model.predict_proba(x_val)[:, 1])
            self.val_auroc_by_c[c] = score
            if best is None or score > best[0]:
                best = (score, c, model)
        assert best is not None
        _, self.chosen_c, self._model = best

    def fit_fixed(self, train: pd.DataFrame, c: float) -> None:
        """Fit with a given C and no tuning (used for the row-split comparison)."""
        x_train = self.embedder.embed(list(train["router_text"]))
        self.chosen_c = c
        self._model = self._new_model(c).fit(x_train, train["label"].astype(int).to_numpy())

    def predict_proba(self, texts: Sequence[str]) -> Scores:
        if self._model is None:
            raise RuntimeError("V0Router.predict_proba called before fit")
        proba: Scores = self._model.predict_proba(self.embedder.embed(texts))[:, 1]
        return proba.astype(np.float64)
