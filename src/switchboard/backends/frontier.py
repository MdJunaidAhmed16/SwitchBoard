"""Client for the frontier model (Claude Opus 5.5 through OpenRouter) with a hard spend guard.

OpenRouter speaks the same chat-completions shape as vLLM, so callers treat both alike
(06-serving). Two rules from 09-security are enforced here:

- The API key is a ``SecretStr`` and never appears in logs, errors or reprs.
- Every request is checked against a per-run dollar ceiling **before** it is sent. The guard
  reserves the request's worst-case cost (prompt bound + the full output budget at the pinned
  prices), so concurrent requests can never push total spend past the limit.

Cost is always computed from the reported token counts and the prices pinned in ``config.py``.
OpenRouter's own reported charge is kept alongside for cross-checking, never used instead.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import SecretStr

from switchboard.backends.base import Generation

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_BACKOFF_S = 30.0


class FrontierError(RuntimeError):
    pass


class SpendLimitError(FrontierError):
    pass


@dataclass
class SpendGuard:
    limit_usd: float
    price_in_per_mtok: float
    price_out_per_mtok: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0

    def cost(self, prompt_tokens: int, output_tokens: int) -> float:
        return (
            prompt_tokens * self.price_in_per_mtok + output_tokens * self.price_out_per_mtok
        ) / 1e6

    def worst_case(self, prompt_chars: int, max_output_tokens: int) -> float:
        # A token is at least ~2 characters of English, so chars / 2 bounds the prompt from above.
        return self.cost(prompt_chars // 2 + 64, max_output_tokens)

    def reserve(self, amount: float) -> None:
        if self.spent_usd + self.reserved_usd + amount > self.limit_usd:
            raise SpendLimitError(
                f"spend guard: ${self.spent_usd:.4f} spent + ${self.reserved_usd:.4f} reserved "
                f"+ ${amount:.4f} for this request would exceed the ${self.limit_usd:.2f} limit"
            )
        self.reserved_usd += amount

    def settle(self, reserved: float, actual: float) -> None:
        self.reserved_usd -= reserved
        self.spent_usd += actual


@dataclass(frozen=True)
class FrontierResult:
    generation: Generation
    cost_usd: float  # tokens x pinned prices
    reported_cost_usd: float | None  # OpenRouter's own figure, for cross-checking only


class FrontierClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: SecretStr,
        guard: SpendGuard,
        timeout_s: float = 300.0,
        max_retries: int = 4,
        max_connections: int = 8,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.guard = guard
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            limits=httpx.Limits(max_connections=max_connections),
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value()}",
                "X-Title": "Switchboard",
            },
            transport=transport,
        )

    async def __aenter__(self) -> FrontierClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def chat(
        self, messages: list[dict[str, str]], decode_params: Mapping[str, Any]
    ) -> FrontierResult:
        prompt_chars = sum(len(m["content"]) for m in messages)
        reserved = self.guard.worst_case(prompt_chars, int(decode_params["max_tokens"]))
        self.guard.reserve(reserved)  # raises before anything is sent
        try:
            result = await self._send(messages, decode_params)
        except BaseException:
            self.guard.settle(reserved, 0.0)
            raise
        self.guard.settle(reserved, result.cost_usd)
        return result

    async def _send(
        self, messages: list[dict[str, str]], decode_params: Mapping[str, Any]
    ) -> FrontierResult:
        payload = {"model": self.model, "messages": messages, "usage": {"include": True},
                   **decode_params}  # fmt: skip
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                response = await self._client.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise FrontierError(f"frontier unreachable: {type(exc).__name__}") from exc
            else:
                if response.status_code == 200:
                    return self._parse(response.json(), (time.perf_counter() - start) * 1000)
                if response.status_code == 402:
                    raise SpendLimitError("OpenRouter: insufficient credits on the key (402)")
                if response.status_code not in _RETRYABLE_STATUS or attempt == self.max_retries:
                    # The body is from the provider, never contains our key; truncate anyway.
                    raise FrontierError(
                        f"frontier returned {response.status_code}: {response.text[:300]}"
                    )
            await asyncio.sleep(min(2.0**attempt, _MAX_BACKOFF_S))
        raise AssertionError("unreachable")

    def _parse(self, body: dict[str, Any], latency_ms: float) -> FrontierResult:
        if "error" in body and "choices" not in body:
            raise FrontierError(f"frontier error: {str(body['error'])[:300]}")
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        reported = usage.get("cost")
        generation = Generation(
            text=(choice.get("message") or {}).get("content") or "",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            finish_reason=str(choice.get("finish_reason") or "unknown"),
        )
        return FrontierResult(
            generation=generation,
            cost_usd=self.guard.cost(prompt_tokens, output_tokens),
            reported_cost_usd=float(reported) if reported is not None else None,
        )
