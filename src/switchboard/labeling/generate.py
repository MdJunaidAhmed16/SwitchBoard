"""Phase 1 label generation: run the local model over each benchmark, grade, store.

    python -m switchboard.labeling.generate --bench gsm8k

For every item: generate once with fixed decoding (cached, resumable), grade with the benchmark's
grader, and merge the rows into ``data/labels/<model>.parquet``. Run metadata (versions, decoding,
vLLM settings, wall-clock) goes to the sidecar ``.meta.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

from switchboard.backends.base import Generation
from switchboard.backends.local import LocalBackendError, VLLMClient
from switchboard.config import REPO_ROOT, Settings, get_settings
from switchboard.labeling.benchmarks import REGISTRY, Item, load_items
from switchboard.labeling.cache import (
    GenerationCache,
    cache_key,
    decode_params_hash,
    prompt_hash,
)
from switchboard.labeling.graders import CODE_BENCHMARKS, grader_for
from switchboard.labeling.sandbox import DockerSandbox
from switchboard.labeling.schema import LABEL_COLUMNS, validate_labels
from switchboard.log import configure_logging, get_logger

log = get_logger(__name__)

# One decode configuration for every benchmark (04-datasets: fixed parameters throughout).
# Greedy, so labels are not sampling noise. A generation cut off at max_tokens is graded as-is:
# failing to answer within the budget is the local model failing.
DECODE_PARAMS: dict[str, Any] = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 1024, "seed": 0}


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def labels_path(settings: Settings) -> Path:
    return settings.labels_dir / f"{model_slug(settings.local_model_id)}.parquet"


def meta_path(settings: Settings) -> Path:
    return settings.labels_dir / f"{model_slug(settings.local_model_id)}.meta.json"


def model_key(settings: Settings) -> str:
    return f"{settings.local_model_id}@{settings.local_model_revision}"


def messages_for(item: Item) -> list[dict[str, str]]:
    return [{"role": "user", "content": item.user_prompt}]


def item_key(item: Item, settings: Settings) -> tuple[str, str, str]:
    p_hash = prompt_hash(messages_for(item))
    d_hash = decode_params_hash(DECODE_PARAMS)
    return cache_key(model_key(settings), p_hash, d_hash), p_hash, d_hash


# --- Generation ------------------------------------------------------------------------------


async def generate_missing(
    items: Sequence[Item],
    client: VLLMClient,
    cache: GenerationCache,
    settings: Settings,
) -> tuple[int, int, list[str]]:
    """Generate every item not already cached. Returns (generated, cache_hits, failed_ids)."""
    todo = [it for it in items if not cache.contains(item_key(it, settings)[0])]
    hits = len(items) - len(todo)
    log.info("generation_plan", total=len(items), cached=hits, to_generate=len(todo))

    semaphore = asyncio.Semaphore(settings.label_concurrency)
    failed: list[str] = []
    done = 0
    start = time.perf_counter()

    async def one(item: Item) -> tuple[Item, Generation | None]:
        async with semaphore:
            try:
                return item, await client.chat(messages_for(item), DECODE_PARAMS)
            except LocalBackendError as exc:
                log.error("generation_failed", item_id=item.item_id, error=str(exc))
                return item, None

    for next_done in asyncio.as_completed([one(it) for it in todo]):
        item, gen = await next_done
        if gen is None:
            failed.append(item.item_id)
            continue
        key, p_hash, d_hash = item_key(item, settings)
        cache.put(key, model_key(settings), p_hash, d_hash, gen)
        done += 1
        if done % 50 == 0:
            cache.commit()
        if done % 200 == 0:
            rate = done / (time.perf_counter() - start)
            log.info("generation_progress", done=done, of=len(todo), items_per_s=round(rate, 2))
    cache.commit()
    return done, hits, failed


# --- Grading ---------------------------------------------------------------------------------


def grade(
    items: Sequence[Item],
    cache: GenerationCache,
    settings: Settings,
    sandbox: DockerSandbox | None,
) -> list[dict[str, Any]]:
    """One label row per cached item. Items without a generation are skipped (and logged)."""
    if not items:
        return []
    benchmark = items[0].benchmark
    if benchmark in CODE_BENCHMARKS and not isinstance(sandbox, DockerSandbox):
        # Model-generated code only ever executes inside the Docker sandbox.
        raise RuntimeError(f"{benchmark} labels require DockerSandbox")
    grader = grader_for(benchmark, sandbox)

    pairs: list[tuple[Item, Generation, str, str]] = []
    for item in items:
        key, p_hash, d_hash = item_key(item, settings)
        gen = cache.get(key)
        if gen is None:
            log.warning("missing_generation", item_id=item.item_id)
            continue
        pairs.append((item, gen, p_hash, d_hash))

    workers = settings.sandbox_workers if benchmark in CODE_BENCHMARKS else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        labels = list(pool.map(lambda p: grader(p[1].text, p[0].reference), pairs))

    return [
        {
            "benchmark": item.benchmark,
            "item_id": item.item_id,
            "prompt_hash": p_hash,
            "router_text": item.router_text,
            "user_prompt": item.user_prompt,
            "reference": item.reference,
            "generation": gen.text,
            "label": bool(label),
            "prompt_tokens": gen.prompt_tokens,
            "output_tokens": gen.output_tokens,
            "latency_ms": gen.latency_ms,
            "finish_reason": gen.finish_reason,
            "model_id": settings.local_model_id,
            "model_revision": settings.local_model_revision,
            "decode_params_hash": d_hash,
        }
        for (item, gen, p_hash, d_hash), label in zip(pairs, labels, strict=True)
    ]


def merge_labels(path: Path, rows: list[dict[str, Any]], benchmarks: set[str]) -> pd.DataFrame:
    """Replace the rows of ``benchmarks`` in the labels file with ``rows``; keep the rest."""
    new = pd.DataFrame(rows, columns=list(LABEL_COLUMNS))
    new = new.astype({"label": bool, "prompt_tokens": "int64", "output_tokens": "int64"})
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old["benchmark"].isin(benchmarks)]
        new = pd.concat([old, new], ignore_index=True)
    new = new.sort_values(["benchmark", "item_id"]).reset_index(drop=True)
    validate_labels(new)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_parquet(path, index=False, compression="zstd")
    return new


# --- Metadata --------------------------------------------------------------------------------


def _git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}{'-dirty' if dirty else ''}"


def update_meta(path: Path, settings: Settings, run: dict[str, Any], extra: dict[str, Any]) -> None:
    meta: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {"runs": []}
    meta.update(
        {
            "model_id": settings.local_model_id,
            "model_revision": settings.local_model_revision,
            "decode_params": DECODE_PARAMS,
            "decode_params_hash": decode_params_hash(DECODE_PARAMS),
            "sandbox_image": settings.sandbox_image,
            **extra,
        }
    )
    meta.setdefault("benchmarks", {})
    bench = REGISTRY[run["benchmark"]]
    meta["benchmarks"][bench.name] = {
        "repo_id": bench.repo_id,
        "revision": bench.revision,
        "license_per_card": bench.license,
        "limit": bench.limit,
        "n_items": run["n_items"],
        "limited_run": run["limited_run"],
    }
    meta["runs"].append(run)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def software_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "switchboard_commit": _git_commit()}
    for pkg in ("pandas", "pyarrow", "huggingface_hub", "httpx"):
        out[pkg] = version(pkg)
    return out


# --- Entry point -----------------------------------------------------------------------------


def _check_docker(settings: Settings) -> DockerSandbox:
    if shutil.which("docker") is None:
        raise SystemExit("docker not found: code benchmarks need the Docker sandbox")
    probe = subprocess.run(
        ["docker", "image", "inspect", settings.sandbox_image], capture_output=True, check=False
    )
    if probe.returncode != 0:
        raise SystemExit("sandbox image not present: run `make sandbox-pull`")
    return DockerSandbox.from_settings(settings)


async def _run(args: argparse.Namespace, settings: Settings) -> int:
    names = list(REGISTRY) if args.bench == "all" else [args.bench]
    sandbox = _check_docker(settings) if CODE_BENCHMARKS & set(names) else None
    cache = GenerationCache(settings.cache_path)
    exit_code = 0

    async with VLLMClient(
        settings.local_base_url,
        settings.local_model_id,
        timeout_s=settings.request_timeout_s,
        max_retries=settings.max_retries,
        max_connections=settings.label_concurrency,
    ) as client:
        served = await client.served_models()
        if settings.local_model_id not in served:
            # Labelling with the wrong model silently invalidates every downstream number.
            raise SystemExit(f"vLLM serves {served}, expected {settings.local_model_id}")
        vllm_version = await client.server_version()

        for name in names:
            items = load_items(name, settings, seed=args.seed)
            if args.limit is not None:
                items = items[: args.limit]
            started = datetime.now(UTC).isoformat()
            t0 = time.perf_counter()
            generated, hits, failed = await generate_missing(items, client, cache, settings)
            wall = time.perf_counter() - t0
            rows = grade(items, cache, settings, sandbox)
            df = merge_labels(labels_path(settings), rows, {name})
            accuracy = sum(r["label"] for r in rows) / max(len(rows), 1)
            update_meta(
                meta_path(settings),
                settings,
                run={
                    "benchmark": name,
                    "started_at": started,
                    "generation_wall_clock_s": round(wall, 1),
                    "generated": generated,
                    "cache_hits": hits,
                    "failed": len(failed),
                    "n_items": len(items),
                    "n_labelled": len(rows),
                    "limited_run": args.limit is not None,
                    "host": platform.node(),
                },
                extra={
                    "vllm": {
                        "version": vllm_version,
                        "gpu_memory_utilization": args.vllm_gpu_util,
                        "max_model_len": args.vllm_max_len,
                        "max_num_seqs": args.vllm_max_seqs,
                    },
                    "software": software_versions(),
                },
            )
            log.info(
                "benchmark_labelled",
                benchmark=name,
                labelled=len(rows),
                failed=len(failed),
                accuracy=round(accuracy, 4),
                total_rows=len(df),
            )
            if failed:
                exit_code = 1
    cache.close()
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--bench", default="all", choices=["all", *REGISTRY])
    parser.add_argument("--limit", type=int, default=None, help="smoke runs only: first N items")
    parser.add_argument("--seed", type=int, default=0)
    # Recorded, not applied: these describe how the vLLM server was launched (see Makefile).
    parser.add_argument("--vllm-gpu-util", type=float, default=None)
    parser.add_argument("--vllm-max-len", type=int, default=None)
    parser.add_argument("--vllm-max-seqs", type=int, default=None)
    args = parser.parse_args(argv)
    configure_logging()
    return asyncio.run(_run(args, get_settings()))


if __name__ == "__main__":
    sys.exit(main())
