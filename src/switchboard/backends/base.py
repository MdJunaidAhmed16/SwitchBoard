"""Types shared by the local and frontier backends."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Generation:
    text: str
    prompt_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str
