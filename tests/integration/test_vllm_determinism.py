"""Needs a live vLLM server (make vllm). Run with: make test-integration"""

import asyncio

import pytest

from switchboard.backends.local import VLLMClient
from switchboard.config import get_settings
from switchboard.labeling.generate import DECODE_PARAMS

pytestmark = pytest.mark.integration

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in "
    "May. How many clips did Natalia sell altogether in April and May?",
    "Write a Python function that returns the n-th Fibonacci number.",
]


async def _twice(prompt: str) -> tuple[str, str]:
    settings = get_settings()
    async with VLLMClient(settings.local_base_url, settings.local_model_id) as client:
        messages = [{"role": "user", "content": prompt}]
        a = await client.chat(messages, DECODE_PARAMS)
        b = await client.chat(messages, DECODE_PARAMS)
        return a.text, b.text


@pytest.mark.parametrize("prompt", PROMPTS)
def test_same_prompt_same_answer_under_fixed_decoding(prompt: str) -> None:
    a, b = asyncio.run(_twice(prompt))
    assert a == b


def test_server_serves_the_configured_model() -> None:
    async def go() -> list[str]:
        settings = get_settings()
        async with VLLMClient(settings.local_base_url, settings.local_model_id) as client:
            return await client.served_models()

    assert get_settings().local_model_id in asyncio.run(go())
