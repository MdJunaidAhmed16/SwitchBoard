"""Per-seed checkpoints for v1, so a long run that is interrupted loses at most one seed.

After a seed is trained, its logit for every labelled prompt, its temperature, best epoch and
training history are saved under ``data/cache/v1/<key>``. The key hashes everything that could
change the result — the labels file, attempt, seed, encoder revision and every hyperparameter —
so a checkpoint is reused only for an identical run.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from switchboard.config import Settings
from switchboard.labeling.generate import labels_path
from switchboard.log import get_logger
from switchboard.router.calibrate import apply_temperature
from switchboard.router.interface import Scores
from switchboard.router.v1 import V1Router, pretrained_backbone

log = get_logger(__name__)

FORMAT_VERSION = "v1-checkpoint-1"


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class V1Scores:
    """A trained v1 seed, reduced to its outputs. Implements the Router interface."""

    seed: int
    temperature: float
    best_epoch: int | None
    history: list[dict[str, float]]
    logits_by_key: dict[str, float] = field(repr=False)
    name: str = field(init=False)

    def __post_init__(self) -> None:
        self.name = f"v1_s{self.seed}"

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        return None  # trained (or loaded) before evaluation

    def logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        return np.array([self.logits_by_key[_text_key(t)] for t in texts], dtype=np.float64)

    def predict_proba_uncalibrated(self, texts: Sequence[str]) -> Scores:
        return apply_temperature(self.logits(texts), 1.0)

    def predict_proba(self, texts: Sequence[str]) -> Scores:
        return apply_temperature(self.logits(texts), self.temperature)


def checkpoint_key(
    settings: Settings, attempt_number: int, seed: int, hyper: dict[str, Any]
) -> str:
    labels_digest = hashlib.sha256(labels_path(settings).read_bytes()).hexdigest()
    payload = {
        "format": FORMAT_VERSION,
        "labels": labels_digest,
        "attempt": attempt_number,
        "seed": seed,
        "encoder": settings.router_encoder_id,
        "revision": settings.router_encoder_revision,
        "hyper": hyper,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def train_or_load(
    settings: Settings,
    attempt_number: int,
    seed: int,
    train: pd.DataFrame,
    val: pd.DataFrame,
    all_texts: Sequence[str],
) -> V1Scores:
    router = V1Router(
        pretrained_backbone(settings.router_encoder_id, settings.router_encoder_revision),
        seed=seed,
        max_tokens=settings.router_max_tokens,
    )
    hyper = {k: getattr(router, k) for k in ("max_epochs", "batch_size", "lr_encoder", "lr_head",
             "weight_decay", "warmup_fraction", "patience", "max_tokens")}  # fmt: skip
    directory = settings.data_dir / "cache" / "v1"
    stem = directory / checkpoint_key(settings, attempt_number, seed, hyper)
    meta_file, logits_file = stem.with_suffix(".json"), stem.with_suffix(".npz")
    if meta_file.exists() and logits_file.exists():
        meta = json.loads(meta_file.read_text())
        arrays = np.load(logits_file)
        log.info("v1_checkpoint_loaded", seed=seed, path=str(stem))
        return V1Scores(
            seed=seed,
            temperature=meta["temperature"],
            best_epoch=meta["best_epoch"],
            history=meta["history"],
            logits_by_key=dict(
                zip(arrays["keys"].tolist(), arrays["logits"].tolist(), strict=True)
            ),
        )

    router.fit(train, val)
    unique = sorted(set(all_texts))
    logits = router.logits(unique)
    scores = V1Scores(
        seed=seed,
        temperature=router.temperature,
        best_epoch=router.best_epoch,
        history=router.history,
        logits_by_key={_text_key(t): float(z) for t, z in zip(unique, logits, strict=True)},
    )
    directory.mkdir(parents=True, exist_ok=True)
    keys = np.array(list(scores.logits_by_key))
    np.savez(logits_file, keys=keys, logits=np.array(list(scores.logits_by_key.values())))
    meta = {k: v for k, v in asdict(scores).items() if k != "logits_by_key"}
    meta_file.write_text(json.dumps({**meta, "hyper": hyper}, indent=2) + "\n")
    log.info("v1_checkpoint_saved", seed=seed, path=str(stem))
    return scores
