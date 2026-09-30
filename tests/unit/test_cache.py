from pathlib import Path

from switchboard.backends.base import Generation
from switchboard.labeling.cache import (
    GenerationCache,
    cache_key,
    decode_params_hash,
    prompt_hash,
)

MESSAGES = [{"role": "user", "content": "What is 2+2?"}]
PARAMS = {"temperature": 0.0, "max_tokens": 16, "seed": 0}
GEN = Generation(text="4", prompt_tokens=5, output_tokens=1, latency_ms=12.5, finish_reason="stop")


def test_same_inputs_same_key() -> None:
    a = cache_key("m@rev", prompt_hash(MESSAGES), decode_params_hash(PARAMS))
    b = cache_key("m@rev", prompt_hash(list(MESSAGES)), decode_params_hash(dict(PARAMS)))
    assert a == b


def test_param_order_does_not_change_key() -> None:
    reordered = {"seed": 0, "max_tokens": 16, "temperature": 0.0}
    assert decode_params_hash(PARAMS) == decode_params_hash(reordered)


def test_key_is_stable_across_processes() -> None:
    # Literals, so a change to the hashing scheme (which would orphan the cache) fails loudly.
    assert prompt_hash(MESSAGES) == (
        "0bd13bfaa6e7191f21633476aedcdcd36b64ad83aa34a06b4eda6d461f1ca108"
    )
    assert decode_params_hash(PARAMS) == (
        "ec028973af5bba9db35b44d665dec614a6796344975c855e76b64ddeb252b7a7"
    )


def test_any_component_change_changes_key() -> None:
    p, d = prompt_hash(MESSAGES), decode_params_hash(PARAMS)
    base = cache_key("m@rev", p, d)
    assert cache_key("m@other", p, d) != base
    assert cache_key("m@rev", prompt_hash([{"role": "user", "content": "What is 2+3?"}]), d) != base
    assert cache_key("m@rev", p, decode_params_hash({**PARAMS, "max_tokens": 17})) != base


def test_roundtrip_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "c.sqlite"
    cache = GenerationCache(path)
    cache.put("k", "m@rev", "p", "d", GEN)
    cache.close()

    reopened = GenerationCache(path)
    assert reopened.contains("k")
    assert reopened.get("k") == GEN
    assert len(reopened) == 1


def test_existing_generation_is_never_overwritten(tmp_path: Path) -> None:
    cache = GenerationCache(tmp_path / "c.sqlite")
    cache.put("k", "m@rev", "p", "d", GEN)
    cache.put("k", "m@rev", "p", "d", Generation("5", 5, 1, 1.0, "stop"))
    assert cache.get("k") == GEN
