"""Frontier client: spend guard, cost accounting and secret handling (no network)."""

import asyncio
import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from switchboard.backends.frontier import (
    FrontierClient,
    FrontierError,
    SpendGuard,
    SpendLimitError,
)

KEY = SecretStr("sk-or-test-not-a-real-key")
MESSAGES = [{"role": "user", "content": "What is 2+2?"}]
PARAMS = {"max_tokens": 1000}


def _ok(prompt: int = 100, completion: int = 50, cost: float | None = 0.00123) -> dict[str, Any]:
    usage: dict[str, Any] = {"prompt_tokens": prompt, "completion_tokens": completion}
    if cost is not None:
        usage["cost"] = cost
    return {"choices": [{"message": {"content": "4"}, "finish_reason": "stop"}], "usage": usage}


def _guard(limit: float = 1.0) -> SpendGuard:
    return SpendGuard(limit_usd=limit, price_in_per_mtok=4.0, price_out_per_mtok=20.0)


def _run(handler: Any, guard: SpendGuard) -> Any:
    async def go() -> Any:
        transport = httpx.MockTransport(handler)
        model = "anthropic/claude-opus-5.5"
        async with FrontierClient(
            "https://fake/api/v1", model, KEY, guard, max_retries=2, transport=transport
        ) as client:
            return await client.chat(MESSAGES, PARAMS)

    return asyncio.run(go())


def test_cost_uses_pinned_prices_not_the_reported_figure() -> None:
    guard = _guard()
    result = _run(lambda r: httpx.Response(200, json=_ok(100, 50, cost=9.99)), guard)
    expected = (100 * 4.0 + 50 * 20.0) / 1e6
    assert result.cost_usd == pytest.approx(expected)
    assert result.reported_cost_usd == 9.99
    assert guard.spent_usd == pytest.approx(expected)
    assert guard.reserved_usd == pytest.approx(0.0)


def test_guard_refuses_before_any_request_is_sent() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=_ok())

    # Worst case for 1,000 output tokens is $0.02 of output alone; a $0.01 limit must refuse.
    with pytest.raises(SpendLimitError, match="exceed"):
        _run(handler, _guard(limit=0.01))
    assert calls == []


def test_concurrent_reservations_cannot_overshoot() -> None:
    guard = _guard(limit=0.05)
    worst = guard.worst_case(len(MESSAGES[0]["content"]), PARAMS["max_tokens"])
    guard.reserve(worst)
    guard.reserve(worst)
    with pytest.raises(SpendLimitError):
        guard.reserve(worst)


def test_failed_request_releases_its_reservation() -> None:
    guard = _guard()
    with pytest.raises(FrontierError):
        _run(lambda r: httpx.Response(400, text="bad request"), guard)
    assert guard.reserved_usd == pytest.approx(0.0)
    assert guard.spent_usd == 0.0


def test_out_of_credits_stops_the_run() -> None:
    with pytest.raises(SpendLimitError, match="402"):
        _run(lambda r: httpx.Response(402, json={"error": "insufficient credits"}), _guard())


def test_rate_limit_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("switchboard.backends.frontier.asyncio.sleep", no_sleep)
    responses = iter([httpx.Response(429), httpx.Response(200, json=_ok())])
    assert _run(lambda r: next(responses), _guard()).generation.text == "4"


def test_interleave_keeps_a_truncated_run_proportional() -> None:
    from switchboard.labeling.benchmarks import Item
    from switchboard.labeling.frontier import interleave

    def items(name: str, n: int) -> list[Item]:
        return [Item(name, f"{name}-{i}", "t", "t", "r") for i in range(n)]

    order = interleave({"bbh": items("bbh", 300), "math": items("math", 100)})
    first = [it.benchmark for it in order[:100]]
    # Any prefix keeps roughly the 3:1 ratio instead of being all one benchmark.
    assert 70 <= first.count("bbh") <= 80
    assert len(order) == 400


def test_key_is_sent_as_bearer_and_never_echoed_in_errors() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(500, text="upstream error")

    with pytest.raises(FrontierError) as err:
        _run(handler, _guard())
    assert seen["auth"] == f"Bearer {KEY.get_secret_value()}"
    assert KEY.get_secret_value() not in str(err.value)
    assert "temperature" not in seen["body"]  # Opus 5.5 rejects sampling parameters
    assert seen["body"]["usage"] == {"include": True}
