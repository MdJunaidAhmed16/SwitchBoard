"""End-to-end label pipeline against a fake vLLM server (httpx.MockTransport)."""

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import pytest

from switchboard.backends.local import LocalBackendError, VLLMClient
from switchboard.config import Settings
from switchboard.labeling.benchmarks import Item
from switchboard.labeling.cache import GenerationCache
from switchboard.labeling.generate import generate_missing, grade, merge_labels
from switchboard.labeling.review import render, sample
from switchboard.labeling.schema import validate_labels
from switchboard.labeling.summary import BEGIN, END, summarise, to_markdown, write_into_report


def _completion(text: str) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }


class FakeVLLM:
    """Always answers 4. Items whose reference is 4 are correct, the rest wrong."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(200, json=_completion("Thinking...\nThe answer is: 4"))


def _items() -> list[Item]:
    return [
        Item("gsm8k", f"test-{i:05d}", f"Q{i}", f"Q{i}\n\nSolve.", "4" if i % 2 == 0 else "5")
        for i in range(6)
    ]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", results_dir=tmp_path / "results",
                    reports_dir=tmp_path / "reports", label_concurrency=3)  # fmt: skip


def _client(handler: Any) -> VLLMClient:
    return VLLMClient("http://fake/v1", "m", transport=httpx.MockTransport(handler))


async def _generate(settings: Settings, fake: FakeVLLM) -> tuple[int, int, list[str]]:
    cache = GenerationCache(settings.cache_path)
    try:
        async with _client(fake) as client:
            return await generate_missing(_items(), client, cache, settings)
    finally:
        cache.close()


def test_generate_grade_merge(settings: Settings) -> None:
    fake = FakeVLLM()
    generated, hits, failed = asyncio.run(_generate(settings, fake))
    assert (generated, hits, failed) == (6, 0, [])

    cache = GenerationCache(settings.cache_path)
    rows = grade(_items(), cache, settings, sandbox=None)
    cache.close()
    path = settings.labels_dir / "m.parquet"
    df = merge_labels(path, rows, {"gsm8k"})
    validate_labels(pd.read_parquet(path))
    assert df["label"].tolist() == [True, False, True, False, True, False]
    assert (df["router_text"] != df["user_prompt"]).all()


def test_rerun_never_regenerates(settings: Settings) -> None:
    fake = FakeVLLM()
    asyncio.run(_generate(settings, fake))
    first_calls = fake.calls
    generated, hits, _ = asyncio.run(_generate(settings, fake))
    assert fake.calls == first_calls
    assert (generated, hits) == (0, 6)


def test_merge_replaces_only_the_rerun_benchmark(settings: Settings) -> None:
    asyncio.run(_generate(settings, FakeVLLM()))
    cache = GenerationCache(settings.cache_path)
    rows = grade(_items(), cache, settings, sandbox=None)
    cache.close()
    path = settings.labels_dir / "m.parquet"
    other = [{**r, "benchmark": "mmlu"} for r in rows]
    merge_labels(path, other, {"mmlu"})
    merge_labels(path, rows, {"gsm8k"})
    df = merge_labels(path, rows[:2], {"gsm8k"})
    assert (df["benchmark"] == "mmlu").sum() == 6
    assert (df["benchmark"] == "gsm8k").sum() == 2


def test_code_benchmark_refuses_non_docker_sandbox(settings: Settings) -> None:
    item = Item("mbpp", "mbpp-0001", "t", "t", json.dumps({"test_setup_code": "", "test_list": []}))
    cache = GenerationCache(settings.cache_path)
    with pytest.raises(RuntimeError, match="DockerSandbox"):
        grade([item], cache, settings, sandbox=None)
    cache.close()


def test_summary_and_review(settings: Settings) -> None:
    asyncio.run(_generate(settings, FakeVLLM()))
    cache = GenerationCache(settings.cache_path)
    df = merge_labels(settings.labels_dir / "m.parquet",
                      grade(_items(), cache, settings, None), {"gsm8k"})  # fmt: skip
    cache.close()
    meta = {"runs": [{"benchmark": "gsm8k", "generation_wall_clock_s": 3.0}], "model_id": "m"}
    summary = summarise(df, {"gsm8k": "train"}, meta)
    assert summary["overall"]["base_rate"] == 0.5
    assert summary["per_benchmark"]["gsm8k"]["flag"] is None
    assert summary["by_role"]["train"]["n"] == 6
    assert "math" in summary["missing_benchmarks"]

    report = settings.reports_dir / "R1.md"
    report.parent.mkdir(parents=True)
    report.write_text(f"# R1\n\n{BEGIN}\nold\n{END}\n\nhand-written notes stay\n")
    write_into_report(report, to_markdown(summary))
    text = report.read_text()
    assert "old" not in text
    assert "Overall base rate: 50.0%" in text
    assert "hand-written notes stay" in text

    picked = sample(df, 4, seed=0)
    assert picked.equals(sample(df, 4, seed=0))
    assert len(picked) == 4
    assert render(picked, 0).count("- [ ] label correct") == 4


# --- Client behaviour --------------------------------------------------------------------------


def test_client_retries_transient_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("switchboard.backends.local.asyncio.sleep", no_sleep)
    responses = iter([httpx.Response(503), httpx.Response(200, json=_completion("ok"))])

    async def go() -> str:
        async with _client(lambda _: next(responses)) as client:
            return (await client.chat([{"role": "user", "content": "x"}], {})).text

    assert asyncio.run(go()) == "ok"


def test_client_does_not_retry_client_errors() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(400, text="bad request")

    async def go() -> None:
        async with _client(handler) as client:
            await client.chat([{"role": "user", "content": "x"}], {})

    with pytest.raises(LocalBackendError, match="400"):
        asyncio.run(go())
    assert len(calls) == 1
