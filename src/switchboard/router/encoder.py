"""Frozen sentence encoder for router v0: CLS embeddings from a pinned checkpoint, cached on disk.

Embeddings are keyed by sha256(text), so each text is encoded once per (model, revision) and the
cache can never serve vectors from a different checkpoint.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from switchboard.config import Settings
from switchboard.log import get_logger

log = get_logger(__name__)

Vectors = NDArray[np.float32]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> Vectors: ...


def _key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class FrozenEncoder:
    def __init__(
        self,
        model_id: str,
        revision: str,
        cache_dir: Path,
        max_tokens: int = 512,
        batch_size: int = 64,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        slug = f"{model_id.replace('/', '__')}@{revision[:12]}"
        self.cache_path = cache_dir / "embeddings" / f"{slug}.parquet"
        # Loaded lazily so importing this module never pulls in torch.
        self._tokenizer: Any = None
        self._model: Any = None

    @classmethod
    def from_settings(cls, settings: Settings) -> FrozenEncoder:
        return cls(
            settings.router_encoder_id,
            settings.router_encoder_revision,
            settings.data_dir / "cache",
            max_tokens=settings.router_max_tokens,
        )

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        model = AutoModel.from_pretrained(self.model_id, revision=self.revision)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        if device == "cuda":
            model = model.half()
        self._model = model
        log.info("encoder_loaded", model=self.model_id, revision=self.revision, device=device)

    def token_lengths(self, texts: Sequence[str]) -> NDArray[np.int64]:
        """Untruncated token count per text, for the truncation-rate report."""
        self._load()
        assert self._tokenizer is not None
        lengths = [len(ids) for ids in self._tokenizer(list(texts), truncation=False)["input_ids"]]
        return np.asarray(lengths, dtype=np.int64)

    def _encode(self, texts: list[str]) -> Vectors:
        import torch

        self._load()
        assert self._tokenizer is not None
        assert self._model is not None
        device = next(self._model.parameters()).device
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(texts), self.batch_size):
                batch = self._tokenizer(
                    texts[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_tokens,
                    return_tensors="pt",
                ).to(device)
                cls = self._model(**batch).last_hidden_state[:, 0]
                cls = torch.nn.functional.normalize(cls.float(), dim=-1)
                chunks.append(cls.cpu().numpy())
        return np.concatenate(chunks).astype(np.float32)

    def embed(self, texts: Sequence[str]) -> Vectors:
        cache: dict[str, Vectors] = {}
        if self.cache_path.exists():
            stored = pd.read_parquet(self.cache_path)
            cache = {
                k: np.asarray(v, dtype=np.float32)
                for k, v in zip(stored["key"], stored["vector"], strict=True)
            }
        keys = [_key(t) for t in texts]
        missing = sorted({k: t for k, t in zip(keys, texts, strict=True) if k not in cache}.items())
        if missing:
            log.info("encoding", new=len(missing), cached=len(set(keys)) - len(missing))
            vectors = self._encode([t for _, t in missing])
            for (k, _), v in zip(missing, vectors, strict=True):
                cache[k] = v
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"key": list(cache), "vector": [v.tolist() for v in cache.values()]}
            ).to_parquet(self.cache_path, index=False)
        return np.stack([cache[k] for k in keys])
