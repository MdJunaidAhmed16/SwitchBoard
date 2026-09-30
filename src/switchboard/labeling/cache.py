"""Generation cache keyed on ``(model_id, prompt_hash, decode_params_hash)``.

Label generation is the largest GPU cost in the project. A rerun must never regenerate a
generation that already exists, and an interrupted run must resume where it stopped. SQLite gives
both with no service to run.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from switchboard.backends.base import Generation


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def prompt_hash(messages: list[dict[str, str]]) -> str:
    return _sha256(_canonical(messages))


def decode_params_hash(params: Mapping[str, Any]) -> str:
    return _sha256(_canonical(dict(params)))


def cache_key(model: str, p_hash: str, d_hash: str) -> str:
    """``model`` is ``<model_id>@<revision>`` so a new checkpoint never reuses old generations."""
    return _sha256(f"{model}|{p_hash}|{d_hash}")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS generations (
    cache_key          TEXT PRIMARY KEY,
    model              TEXT NOT NULL,
    prompt_hash        TEXT NOT NULL,
    decode_params_hash TEXT NOT NULL,
    text               TEXT NOT NULL,
    prompt_tokens      INTEGER NOT NULL,
    output_tokens      INTEGER NOT NULL,
    latency_ms         REAL NOT NULL,
    finish_reason      TEXT NOT NULL,
    created_at         TEXT NOT NULL
)
"""


class GenerationCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def get(self, key: str) -> Generation | None:
        row = self._conn.execute(
            "SELECT text, prompt_tokens, output_tokens, latency_ms, finish_reason "
            "FROM generations WHERE cache_key = ?",
            (key,),
        ).fetchone()
        return Generation(*row) if row else None

    def contains(self, key: str) -> bool:
        return (
            self._conn.execute("SELECT 1 FROM generations WHERE cache_key = ?", (key,)).fetchone()
            is not None
        )

    def put(self, key: str, model: str, p_hash: str, d_hash: str, gen: Generation) -> None:
        # INSERT OR IGNORE: an existing generation is never overwritten.
        self._conn.execute(
            "INSERT OR IGNORE INTO generations VALUES "
            "(:key, :model, :p, :d, :text, :prompt_tokens, :output_tokens, :latency_ms, "
            ":finish_reason, :created_at)",
            {
                "key": key,
                "model": model,
                "p": p_hash,
                "d": d_hash,
                **asdict(gen),
                "created_at": datetime.now(UTC).isoformat(),
            },
        )

    def commit(self) -> None:
        self._conn.commit()

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __len__(self) -> int:
        count: int = self._conn.execute("SELECT COUNT(*) FROM generations").fetchone()[0]
        return count
