"""Dataset-level split: which benchmark is train, val or test (``data/splits.yaml``)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

Role = Literal["train", "val", "test"]
ROLES: tuple[Role, ...] = ("train", "val", "test")


def load_splits(path: Path, known: set[str] | None = None) -> dict[str, Role]:
    """Map benchmark name -> role, refusing overlaps and unknown names."""
    raw = yaml.safe_load(path.read_text())
    if set(raw) != set(ROLES):
        raise ValueError(f"splits must define exactly {ROLES}, got {sorted(raw)}")
    roles: dict[str, Role] = {}
    for role in ROLES:
        for name in raw[role]:
            if name in roles:
                raise ValueError(f"{name!r} is in both {roles[name]!r} and {role!r}")
            roles[name] = role
    if known is not None and (unknown := set(roles) - known):
        raise ValueError(f"unknown benchmarks in splits: {sorted(unknown)}")
    return roles
