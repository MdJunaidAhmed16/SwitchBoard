"""Benchmark registry: where each dataset comes from, how it is sampled, how it is prompted.

Every source is pinned to a Hugging Face commit hash. Sampling is a deterministic hash order, so
the same seed selects the same rows regardless of library versions.

Two texts exist per item, on purpose:

- ``router_text`` is the task itself. It is what the router sees.
- ``user_prompt`` is ``router_text`` plus a per-track answer-format instruction. It is what the
  local model sees. The instruction differs by track, so letting the router see it would let it
  learn benchmark provenance instead of difficulty.
"""

from __future__ import annotations

import functools
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from switchboard.config import Settings


@dataclass(frozen=True)
class Item:
    benchmark: str
    item_id: str
    router_text: str
    user_prompt: str
    reference: str


@dataclass(frozen=True)
class Benchmark:
    name: str
    track: str
    repo_id: str
    revision: str
    files: tuple[str, ...]
    license: str
    build: Callable[[pd.DataFrame], list[Item]]
    limit: int | None = None


# --- Answer-format instructions (generation only, never shown to the router) --------------------

NUMERIC_INSTRUCTION = (
    "Solve the problem step by step. "
    'Finish with a final line of the form "The answer is: <number>".'
)
MATH_INSTRUCTION = "Solve the problem step by step. Put your final answer within \\boxed{}."
CHOICE_INSTRUCTION = (
    "Think it through briefly, then finish with a final line of the form "
    '"The answer is (X)", where X is the letter of the correct option.'
)
BBH_INSTRUCTION = (
    'Think step by step, then finish with a final line of the form "The answer is <answer>".'
)
MBPP_INSTRUCTION = "Write the Python function in a single ```python code block."


def _with_instruction(task: str, instruction: str) -> str:
    return f"{task}\n\n{instruction}"


def _choices_block(options: list[str]) -> str:
    return "\n".join(f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options))


def hash_sample(items: list[Item], n: int | None, seed: int) -> list[Item]:
    """Deterministic subset: order by sha256(seed:item_id), take the first n."""
    if n is None or len(items) <= n:
        return sorted(items, key=lambda it: it.item_id)
    keyed = sorted(
        items, key=lambda it: hashlib.sha256(f"{seed}:{it.item_id}".encode()).hexdigest()
    )
    return sorted(keyed[:n], key=lambda it: it.item_id)


# --- Builders: DataFrame of raw rows -> Items ----------------------------------------------------


def build_gsm8k(df: pd.DataFrame) -> list[Item]:
    items = []
    for split, idx, question, answer in zip(
        df["_split"], df["_row"], df["question"], df["answer"], strict=True
    ):
        items.append(
            Item(
                benchmark="gsm8k",
                item_id=f"{split}-{idx:05d}",
                router_text=question,
                user_prompt=_with_instruction(question, NUMERIC_INSTRUCTION),
                reference=answer.rsplit("####", 1)[1].strip().replace(",", ""),
            )
        )
    return items


def build_math(df: pd.DataFrame) -> list[Item]:
    return [
        Item(
            benchmark="math",
            item_id=uid,
            router_text=problem,
            user_prompt=_with_instruction(problem, MATH_INSTRUCTION),
            reference=answer,
        )
        for uid, problem, answer in zip(df["unique_id"], df["problem"], df["answer"], strict=True)
    ]


def build_mmlu(df: pd.DataFrame) -> list[Item]:
    items = []
    for idx, question, choices, answer in zip(
        df["_row"], df["question"], df["choices"], df["answer"], strict=True
    ):
        task = f"{question}\n\n{_choices_block(list(choices))}"
        items.append(
            Item(
                benchmark="mmlu",
                item_id=f"test-{idx:05d}",
                router_text=task,
                user_prompt=_with_instruction(task, CHOICE_INSTRUCTION),
                reference="ABCDEFGHIJ"[int(answer)],
            )
        )
    return items


def build_arc(df: pd.DataFrame) -> list[Item]:
    items = []
    for item_id, question, choices, answer_key in zip(
        df["id"], df["question"], df["choices"], df["answerKey"], strict=True
    ):
        labels = list(choices["label"])
        texts = list(choices["text"])
        # A few ARC rows label options "1".."4"; options are re-lettered by position.
        position = labels.index(answer_key)
        task = f"{question}\n\n{_choices_block(texts)}"
        items.append(
            Item(
                benchmark="arc_challenge",
                item_id=str(item_id),
                router_text=task,
                user_prompt=_with_instruction(task, CHOICE_INSTRUCTION),
                reference="ABCDEFGHIJ"[position],
            )
        )
    return items


BBH_PER_TASK = 60


def build_bbh(df: pd.DataFrame) -> list[Item]:
    """Stratified: the same number of items from each of the 27 tasks."""
    items: list[Item] = []
    for task, group in df.groupby("_file", sort=True):
        task_name = Path(str(task)).parent.name
        task_items = [
            Item(
                benchmark="bbh",
                item_id=f"{task_name}-{idx:03d}",
                router_text=text,
                user_prompt=_with_instruction(text, BBH_INSTRUCTION),
                reference=target,
            )
            for idx, text, target in zip(
                group["_row"], group["input"], group["target"], strict=True
            )
        ]
        items.extend(hash_sample(task_items, BBH_PER_TASK, seed=0))
    return items


def build_humaneval(df: pd.DataFrame) -> list[Item]:
    items = []
    for task_id, prompt, test, entry in zip(
        df["task_id"], df["prompt"], df["test"], df["entry_point"], strict=True
    ):
        items.append(
            Item(
                benchmark="humaneval",
                item_id=task_id,
                router_text=prompt,
                user_prompt=(
                    "Complete the following Python function. Reply with the full function, "
                    "including the signature, in a single ```python code block.\n\n"
                    f"```python\n{prompt}```"
                ),
                reference=json.dumps({"prompt": prompt, "test": test, "entry_point": entry}),
            )
        )
    return items


def build_mbpp(df: pd.DataFrame) -> list[Item]:
    items = []
    for task_id, text, tests, setup in zip(
        df["task_id"], df["text"], df["test_list"], df["test_setup_code"], strict=True
    ):
        test_list = list(tests)
        task = f"{text}\nYour code should pass these tests:\n" + "\n".join(test_list)
        items.append(
            Item(
                benchmark="mbpp",
                item_id=f"mbpp-{int(task_id):04d}",
                router_text=task,
                user_prompt=_with_instruction(task, MBPP_INSTRUCTION),
                reference=json.dumps({"test_setup_code": setup, "test_list": test_list}),
            )
        )
    return items


_BBH_TASKS = (
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
)

# Licences are as stated on each source's dataset card. Verify at source before publishing.
REGISTRY: dict[str, Benchmark] = {
    b.name: b
    for b in [
        Benchmark(
            name="gsm8k",
            track="grade-school maths",
            repo_id="openai/gsm8k",
            revision="740312add88f781978c0658806c59bc2815b9866",
            files=("main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet"),
            license="MIT",
            build=build_gsm8k,
            limit=3000,
        ),
        Benchmark(
            name="math",
            track="competition maths",
            repo_id="HuggingFaceH4/MATH-500",
            revision="6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be",
            files=("test.jsonl",),
            license="MIT (MATH); verify MATH-500 card",
            build=build_math,
        ),
        Benchmark(
            name="mmlu",
            track="broad knowledge",
            repo_id="cais/mmlu",
            revision="c30699e8356da336a370243923dbaf21066bb9fe",
            files=("all/test-00000-of-00001.parquet",),
            license="MIT",
            build=build_mmlu,
            limit=3000,
        ),
        Benchmark(
            name="arc_challenge",
            track="science reasoning (validation)",
            repo_id="allenai/ai2_arc",
            revision="210d026faf9955653af8916fad021475a3f00453",
            files=("ARC-Challenge/test-00000-of-00001.parquet",),
            license="CC-BY-SA-4.0",
            build=build_arc,
        ),
        Benchmark(
            name="bbh",
            track="multi-step reasoning",
            repo_id="lukaemon/bbh",
            revision="982bb89fd79532a8ac676a61fc42eb1aeec63f99",
            files=tuple(f"{t}/test-00000-of-00001.parquet" for t in _BBH_TASKS),
            license="MIT (BIG-Bench Hard); verify mirror card",
            build=build_bbh,
        ),
        Benchmark(
            name="humaneval",
            track="code generation",
            repo_id="openai/openai_humaneval",
            revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
            files=("openai_humaneval/test-00000-of-00001.parquet",),
            license="MIT",
            build=build_humaneval,
        ),
        Benchmark(
            name="mbpp",
            track="code generation",
            repo_id="google-research-datasets/mbpp",
            revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
            files=tuple(
                f"full/{s}-00000-of-00001.parquet"
                for s in ("train", "test", "validation", "prompt")
            ),
            license="CC-BY-4.0",
            build=build_mbpp,
        ),
    ]
}


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_parquet(path)


_DOWNLOAD_ATTEMPTS = 4


def _fetch(benchmark: Benchmark, filename: str, settings: Settings) -> Path:
    """Pinned file from the local cache if present, else from the Hub with retries."""
    from huggingface_hub import hf_hub_download

    download = functools.partial(
        hf_hub_download,
        repo_id=benchmark.repo_id,
        filename=filename,
        repo_type="dataset",
        revision=benchmark.revision,
        cache_dir=settings.raw_dir,
    )
    try:
        return Path(download(local_files_only=True))
    except Exception:  # noqa: BLE001 - not cached yet; fall through to a download
        pass
    for attempt in range(1, _DOWNLOAD_ATTEMPTS + 1):
        try:
            return Path(download())
        except Exception:
            if attempt == _DOWNLOAD_ATTEMPTS:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def load_raw(benchmark: Benchmark, settings: Settings) -> pd.DataFrame:
    """Download (or reuse) the pinned files and concatenate them with provenance columns."""
    frames = []
    for filename in benchmark.files:
        frame = _read(_fetch(benchmark, filename, settings)).reset_index(drop=True)
        frame["_file"] = filename
        frame["_split"] = Path(filename).name.split("-")[0]
        frame["_row"] = range(len(frame))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_items(name: str, settings: Settings, seed: int = 0) -> list[Item]:
    benchmark = REGISTRY[name]
    items = benchmark.build(load_raw(benchmark, settings))
    ids = [it.item_id for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate item ids")
    return hash_sample(items, benchmark.limit, seed)
