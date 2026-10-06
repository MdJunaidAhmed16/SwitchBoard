"""Router v1: the v0 encoder fine-tuned end to end (05-router-model, Training).

Same ``bge-base-en-v1.5`` backbone as v0, so the gain over v0 is the value of fine-tuning alone.
A linear head on the CLS vector, trained with (optionally weighted) BCE:

    AdamW · encoder lr 2e-5 · head lr 1e-3 · weight decay 0.01 · 10% linear warm-up, then linear
    decay · up to 4 epochs · early stop on validation AUROC (patience 1) · batch 32 · bf16 on GPU

The best epoch by validation AUROC is kept. A temperature is then fitted on validation logits only
(``calibrate.py``); ``predict_proba`` returns calibrated probabilities.
"""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from switchboard.log import get_logger
from switchboard.router.calibrate import apply_temperature, fit_temperature
from switchboard.router.interface import Scores, sample_weights
from switchboard.router.metrics import auroc

log = get_logger(__name__)

# (tokenizer, encoder) factory; injectable so tests can use a tiny local model.
Backbone = Callable[[], tuple[Any, Any]]


def pretrained_backbone(model_id: str, revision: str) -> Backbone:
    def load() -> tuple[Any, Any]:
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        return tokenizer, AutoModel.from_pretrained(model_id, revision=revision)

    return load


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class V1Router:
    def __init__(
        self,
        backbone: Backbone,
        seed: int = 0,
        max_epochs: int = 4,
        batch_size: int = 32,
        lr_encoder: float = 2e-5,
        lr_head: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_fraction: float = 0.1,
        patience: int = 1,
        max_tokens: int = 512,
        device: str | None = None,
    ) -> None:
        self.name = f"v1_s{seed}"
        self.backbone = backbone
        self.seed = seed
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr_encoder = lr_encoder
        self.lr_head = lr_head
        self.weight_decay = weight_decay
        self.warmup_fraction = warmup_fraction
        self.patience = patience
        self.max_tokens = max_tokens
        self.device = device
        self.temperature: float = 1.0
        self.history: list[dict[str, float]] = []
        self.best_epoch: int | None = None
        self._tokenizer: Any = None
        self._model: Any = None

    # --- model ----------------------------------------------------------------------------------

    def _build(self) -> None:
        import torch

        _seed_everything(self.seed)
        tokenizer, encoder = self.backbone()
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")

        class Classifier(torch.nn.Module):
            def __init__(self, encoder: Any) -> None:
                super().__init__()
                self.encoder = encoder
                self.dropout = torch.nn.Dropout(0.1)
                self.head = torch.nn.Linear(encoder.config.hidden_size, 1)

            def forward(self, **batch: Any) -> Any:
                cls = self.encoder(**batch).last_hidden_state[:, 0]
                return self.head(self.dropout(cls)).squeeze(-1)

        self._tokenizer = tokenizer
        self._model = Classifier(encoder).to(device)

    def _batches(self, texts: Sequence[str]) -> Any:
        device = next(self._model.parameters()).device
        for start in range(0, len(texts), self.batch_size):
            yield self._tokenizer(
                list(texts[start : start + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=self.max_tokens,
                return_tensors="pt",
            ).to(device)

    def logits(self, texts: Sequence[str]) -> NDArray[np.float64]:
        import torch

        if self._model is None:
            raise RuntimeError("V1Router used before fit")
        self._model.eval()
        out = []
        use_bf16 = next(self._model.parameters()).device.type == "cuda"
        with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16, enabled=use_bf16):
            for batch in self._batches(texts):
                out.append(self._model(**batch).float().cpu().numpy())
        return np.concatenate(out).astype(np.float64)

    # --- training -------------------------------------------------------------------------------

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> None:
        import torch

        self._build()
        model = self._model
        texts = list(train["router_text"])
        labels = train["label"].to_numpy(dtype=np.float32)
        weights = sample_weights(train)
        w = np.ones(len(texts), np.float32) if weights is None else weights.astype(np.float32)
        val_texts, val_y = list(val["router_text"]), val["label"].to_numpy(dtype=int)

        head_params = list(model.head.parameters())
        head_ids = {id(p) for p in head_params}
        encoder_params = [p for p in model.parameters() if id(p) not in head_ids]
        optimizer = torch.optim.AdamW(
            [{"params": encoder_params, "lr": self.lr_encoder},
             {"params": head_params, "lr": self.lr_head}],
            weight_decay=self.weight_decay,
        )  # fmt: skip
        steps_per_epoch = math.ceil(len(texts) / self.batch_size)
        total = steps_per_epoch * self.max_epochs
        warmup = max(1, int(total * self.warmup_fraction))

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            return max(0.0, (total - step) / max(1, total - warmup))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")
        device = next(model.parameters()).device
        use_bf16 = device.type == "cuda"
        rng = np.random.default_rng(self.seed)

        best_auc, best_state, stale = -1.0, None, 0
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            order = rng.permutation(len(texts))
            running = 0.0
            for start in range(0, len(order), self.batch_size):
                idx = order[start : start + self.batch_size]
                batch = self._tokenizer(
                    [texts[i] for i in idx],
                    padding=True,
                    truncation=True,
                    max_length=self.max_tokens,
                    return_tensors="pt",
                ).to(device)
                y = torch.tensor(labels[idx], device=device)
                bw = torch.tensor(w[idx], device=device)
                with torch.autocast("cuda", torch.bfloat16, enabled=use_bf16):
                    z = model(**batch).float()
                loss = (loss_fn(z, y) * bw).sum() / bw.sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                running += float(loss) * len(idx)
            val_auc = auroc(val_y, self.logits(val_texts))
            self.history.append(
                {"epoch": epoch, "train_loss": running / len(texts), "val_auroc": val_auc}
            )
            log.info("v1_epoch", seed=self.seed, epoch=epoch, val_auroc=round(val_auc, 4),
                     train_loss=round(running / len(texts), 4))  # fmt: skip
            if val_auc > best_auc:
                best_auc, best_state, stale = val_auc, copy.deepcopy(model.state_dict()), 0
                self.best_epoch = epoch
            else:
                stale += 1
                if stale > self.patience:
                    break
        assert best_state is not None
        model.load_state_dict(best_state)
        # Calibration uses validation only; test never influences any fitted value.
        self.temperature = fit_temperature(self.logits(val_texts), val_y)

    def predict_proba_uncalibrated(self, texts: Sequence[str]) -> Scores:
        return apply_temperature(self.logits(texts), 1.0)

    def predict_proba(self, texts: Sequence[str]) -> Scores:
        return apply_temperature(self.logits(texts), self.temperature)
