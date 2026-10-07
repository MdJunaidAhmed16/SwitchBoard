"""Cost model (08-evaluation). Every cost is tokens x a pinned price, never an estimate.

- Frontier: prompt and output tokens actually reported for each answer x the pinned OpenRouter
  prices for Claude Opus 5.5.
- Local: prompt and output tokens recorded at labelling time x the pinned market price of serving
  a small open model of the same family (see ``config.py`` and the R2 amendment). Scaling this
  price is the sensitivity check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from switchboard.config import Settings


@dataclass(frozen=True)
class Prices:
    in_per_mtok: float
    out_per_mtok: float

    def cost(self, prompt_tokens: ArrayLike, output_tokens: ArrayLike) -> NDArray[np.float64]:
        p = np.asarray(prompt_tokens, dtype=np.float64)
        o = np.asarray(output_tokens, dtype=np.float64)
        return (p * self.in_per_mtok + o * self.out_per_mtok) / 1e6


def frontier_prices(settings: Settings) -> Prices:
    return Prices(settings.frontier_price_in_per_mtok, settings.frontier_price_out_per_mtok)


def local_prices(settings: Settings, scale: float = 1.0) -> Prices:
    return Prices(
        settings.local_price_in_per_mtok * scale, settings.local_price_out_per_mtok * scale
    )
