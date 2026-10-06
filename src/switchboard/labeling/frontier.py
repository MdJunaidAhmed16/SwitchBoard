"""Frontier answers for the held-out test prompts: Claude Opus 5.5 through OpenRouter.

    python -m switchboard.labeling.frontier --pilot 10      (make frontier-pilot)
    python -m switchboard.labeling.frontier --max-usd 25    (make frontier MAX_USD=25)

The cost-quality curve needs, for every test prompt, whether the frontier model answers it
correctly. Prompts, graders and the labels schema are exactly those of the local model, so the
two are compared like for like. Every answer is cached: a re-run never pays twice.

``--pilot N`` answers N prompts per test benchmark (a fixed, seeded subset), then projects the
cost of the full run from the measured spend per item. Nothing beyond the pilot is spent until a
full run is started with an explicit ``--max-usd``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from switchboard.backends.frontier import (
    FrontierClient,
    FrontierError,
    SpendGuard,
    SpendLimitError,
)
from switchboard.config import Settings, get_settings
from switchboard.labeling.benchmarks import REGISTRY, Item, hash_sample, load_items
from switchboard.labeling.cache import GenerationCache
from switchboard.labeling.generate import (
    ModelSpec,
    _check_docker,
    grade,
    item_key,
    merge_labels,
    messages_for,
    model_slug,
)
from switchboard.labeling.graders import CODE_BENCHMARKS
from switchboard.labeling.splits import load_splits
from switchboard.log import configure_logging, get_logger

log = get_logger(__name__)

# Opus 5.5 always thinks and accepts no sampling parameters, so only the output budget is set.
# The budget covers thinking plus the answer; it is generous so the frontier is not truncated.
FRONTIER_DECODE_PARAMS: dict[str, Any] = {"max_tokens": 8192}


def frontier_spec(settings: Settings) -> ModelSpec:
    # API models have no checkpoint hash; the provider is recorded instead.
    return ModelSpec(settings.frontier_model_id, "openrouter", FRONTIER_DECODE_PARAMS)


def held_out_items(settings: Settings, pilot: int | None) -> dict[str, list[Item]]:
    roles = load_splits(settings.splits_path, known=set(REGISTRY))
    names = sorted(b for b, r in roles.items() if r == "test")
    out = {}
    for name in names:
        items = load_items(name, settings)
        out[name] = hash_sample(items, pilot, seed=settings.seed) if pilot else items
    return out


async def answer(
    items: Sequence[Item], settings: Settings, guard: SpendGuard, cache: GenerationCache
) -> dict[str, Any]:
    spec = frontier_spec(settings)
    if settings.openrouter_api_key is None:
        raise SystemExit("OPENROUTER_API_KEY is not set (put it in the gitignored .env)")
    todo = [it for it in items if not cache.contains(item_key(it, settings, spec)[0])]
    stats: dict[str, Any] = {"cached": len(items) - len(todo), "answered": 0, "failed": 0,
                             "stopped_by_spend_guard": False, "reported_cost_usd": 0.0,
                             "cost_usd_by_benchmark": {}, "answered_by_benchmark": {}}  # fmt: skip
    semaphore = asyncio.Semaphore(settings.frontier_concurrency)
    stop = asyncio.Event()

    async with FrontierClient(
        settings.frontier_base_url,
        spec.model_id,
        settings.openrouter_api_key,
        guard,
        timeout_s=settings.request_timeout_s,
        max_retries=settings.max_retries,
        max_connections=settings.frontier_concurrency,
    ) as client:

        async def one(item: Item) -> None:
            async with semaphore:
                if stop.is_set():
                    return
                try:
                    result = await client.chat(messages_for(item), spec.decode_params)
                except SpendLimitError as exc:
                    stats["stopped_by_spend_guard"] = True
                    stop.set()
                    log.warning("spend_guard_stop", reason=str(exc))
                    return
                except FrontierError as exc:
                    stats["failed"] += 1
                    log.error("frontier_failed", item_id=item.item_id, error=str(exc))
                    return
                key, p_hash, d_hash = item_key(item, settings, spec)
                cache.put(key, spec.key, p_hash, d_hash, result.generation)
                cache.commit()
                b = item.benchmark
                stats["answered"] += 1
                stats["answered_by_benchmark"][b] = stats["answered_by_benchmark"].get(b, 0) + 1
                stats["cost_usd_by_benchmark"][b] = (
                    stats["cost_usd_by_benchmark"].get(b, 0.0) + result.cost_usd
                )
                stats["reported_cost_usd"] += result.reported_cost_usd or 0.0
                if stats["answered"] % 50 == 0:
                    log.info("frontier_progress", answered=stats["answered"], of=len(todo),
                             spent_usd=round(guard.spent_usd, 4))  # fmt: skip

        await asyncio.gather(*(one(it) for it in todo))
    stats["spent_usd"] = guard.spent_usd
    return stats


def run(settings: Settings, pilot: int | None, max_usd: float) -> dict[str, Any]:
    spec = frontier_spec(settings)
    by_bench = held_out_items(settings, pilot)
    items = [it for group in by_bench.values() for it in group]
    guard = SpendGuard(max_usd, settings.frontier_price_in_per_mtok,
                       settings.frontier_price_out_per_mtok)  # fmt: skip
    cache = GenerationCache(settings.cache_path)
    started = time.perf_counter()
    stats = asyncio.run(answer(items, settings, guard, cache))
    sandbox = _check_docker(settings) if CODE_BENCHMARKS & set(by_bench) else None

    labels_file = settings.labels_dir / f"{model_slug(spec.model_id)}.parquet"
    summary: dict[str, Any] = {"benchmarks": {}}
    for name, group in by_bench.items():
        rows = grade(group, cache, settings, sandbox, spec=spec)
        if rows:
            merge_labels(labels_file, rows, {name})
        n_cost = len(rows)
        summary["benchmarks"][name] = {
            "items_in_run": len(group),
            "labelled": n_cost,
            "correct": sum(r["label"] for r in rows),
            "accuracy": sum(r["label"] for r in rows) / n_cost if n_cost else None,
            "mean_output_tokens": sum(r["output_tokens"] for r in rows) / n_cost
            if n_cost
            else None,
            "mean_prompt_tokens": sum(r["prompt_tokens"] for r in rows) / n_cost
            if n_cost
            else None,
        }
    cache.close()

    # Cost per item from what is cached (tokens x pinned prices), so a resumed run still projects.
    full = held_out_items(settings, None)
    projection = {}
    for name, b in summary["benchmarks"].items():
        if b["labelled"]:
            per_item = guard.cost(round(b["mean_prompt_tokens"]), round(b["mean_output_tokens"]))
            projection[name] = {"per_item_usd": per_item, "items": len(full[name]),
                                "projected_usd": per_item * len(full[name])}  # fmt: skip
    summary.update({
        "model": spec.model_id, "provider": "openrouter", "decode_params": spec.decode_params,
        "prices_per_mtok": {"in": settings.frontier_price_in_per_mtok,
                            "out": settings.frontier_price_out_per_mtok},
        "mode": f"pilot ({pilot} per benchmark)" if pilot else "full",
        "max_usd": max_usd, "run": stats, "wall_clock_s": round(time.perf_counter() - started, 1),
        "projection_full_test_set": projection,
        "projected_total_usd": sum(p["projected_usd"] for p in projection.values()),
        "finished_at": datetime.now(UTC).isoformat(),
    })  # fmt: skip
    out = settings.results_dir / ("frontier-pilot.json" if pilot else "frontier-run.json")
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    log.info(
        "frontier_done",
        spent_usd=round(stats["spent_usd"], 4),
        projected_total_usd=round(summary["projected_total_usd"], 2),
        path=str(out),
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--pilot", type=int, default=None, help="answer N prompts per benchmark")
    parser.add_argument("--max-usd", type=float, default=None, help="hard ceiling for this run")
    args = parser.parse_args(argv)
    configure_logging()
    settings = get_settings()
    summary = run(settings, args.pilot, args.max_usd or settings.frontier_max_usd_per_run)
    return 1 if summary["run"]["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
