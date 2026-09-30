"""Client for the local model, served by vLLM's OpenAI-compatible server.

The gateway and the labeller treat vLLM as an opaque endpoint. This client speaks the same
chat-completions shape the frontier client will, so the two are interchangeable to callers.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from typing import Any

import httpx

from switchboard.backends.base import Generation

# Retry only what can succeed on a second attempt.
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
_MAX_BACKOFF_S = 30.0


class LocalBackendError(RuntimeError):
    pass


class VLLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 300.0,
        max_retries: int = 4,
        max_connections: int = 32,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=timeout_s,
            limits=httpx.Limits(max_connections=max_connections),
            transport=transport,
        )

    async def __aenter__(self) -> VLLMClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    async def served_models(self) -> list[str]:
        response = await self._client.get(f"{self.base_url}/models")
        response.raise_for_status()
        return [m["id"] for m in response.json()["data"]]

    async def server_version(self) -> str | None:
        root = self.base_url.removesuffix("/v1")
        try:
            response = await self._client.get(f"{root}/version")
            response.raise_for_status()
            version: str = response.json()["version"]
            return version
        except (httpx.HTTPError, KeyError, ValueError):
            return None

    async def chat(
        self, messages: list[dict[str, str]], decode_params: Mapping[str, Any]
    ) -> Generation:
        payload = {"model": self.model, "messages": messages, **decode_params}
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                response = await self._client.post(
                    f"{self.base_url}/chat/completions", json=payload
                )
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise LocalBackendError(f"vLLM unreachable: {exc!r}") from exc
            else:
                if response.status_code == 200:
                    latency_ms = (time.perf_counter() - start) * 1000
                    body = response.json()
                    choice = body["choices"][0]
                    usage = body["usage"]
                    return Generation(
                        text=choice["message"]["content"] or "",
                        prompt_tokens=int(usage["prompt_tokens"]),
                        output_tokens=int(usage["completion_tokens"]),
                        latency_ms=latency_ms,
                        finish_reason=str(choice.get("finish_reason") or "unknown"),
                    )
                if response.status_code not in _RETRYABLE_STATUS or attempt == self.max_retries:
                    raise LocalBackendError(
                        f"vLLM returned {response.status_code}: {response.text[:500]}"
                    )
            await asyncio.sleep(min(2.0**attempt, _MAX_BACKOFF_S))
        raise AssertionError("unreachable")
