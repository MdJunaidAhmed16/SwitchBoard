# Switchboard — every sanctioned entry point lives here.
# Runs on Linux / WSL2. See the phase targets below for what each stage needs.

SHELL := /bin/bash
.DEFAULT_GOAL := help

UV       ?= uv
RUN      := $(UV) run
BENCH    ?= all
LIMIT    ?=
SEED     ?= 0

# vLLM runs as an external server in its own tool environment, so its pinned torch/CUDA never
# collides with the project's training dependencies.
VLLM_VERSION   ?= 0.30.0
VLLM_PORT      ?= 8001
VLLM_GPU_UTIL  ?= 0.85
VLLM_MAX_LEN   ?= 4096
VLLM_MAX_SEQS  ?= 32
# float16 for AWQ (4-bit) checkpoints
VLLM_DTYPE     ?= bfloat16
LOCAL_MODEL    ?= $(shell $(RUN) python -c "from switchboard.config import get_settings as g; print(g().local_model_id)")
LOCAL_REVISION ?= $(shell $(RUN) python -c "from switchboard.config import get_settings as g; print(g().local_model_revision)")

.PHONY: help install lint format typecheck test test-docker test-integration check \
        sandbox-pull grader-selfcheck vllm labels regrade labels-summary review-sample train-v0 train-attempt2 frontier-pilot frontier sweep readme

help:  ## Show targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

# --- Development ---------------------------------------------------------------------------

install:  ## Create the env from uv.lock and install git hooks
	$(UV) sync --frozen
	$(RUN) pre-commit install

lint:  ## Ruff lint + format check
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

format:  ## Apply ruff formatting and safe fixes
	$(RUN) ruff check --fix src tests
	$(RUN) ruff format src tests

typecheck:  ## mypy strict on src/
	$(RUN) mypy

test:  ## Unit tests (no network, no GPU, no Docker)
	$(RUN) pytest

test-docker:  ## Sandbox + code-grader tests that execute inside Docker
	$(RUN) pytest -m docker

test-integration:  ## Tests against a live vLLM server on $(VLLM_PORT)
	$(RUN) pytest -m integration

check: lint typecheck test  ## Everything CI runs on a pull request

# --- Phase 1: labels -------------------------------------------------------------------------

sandbox-pull:  ## Pull the digest-pinned sandbox image used by the code graders
	docker pull $$($(RUN) python -c "from switchboard.config import get_settings as g; print(g().sandbox_image)")

grader-selfcheck:  ## Oracle answers for every real item must grade True (code runs in Docker)
	$(RUN) python -m switchboard.labeling.selfcheck --bench all

vllm:  ## Serve the local model with vLLM on the GPU (foreground)
	$(UV) tool run --python 3.11 --from 'vllm==$(VLLM_VERSION)' vllm serve $(LOCAL_MODEL) \
	  --revision $(LOCAL_REVISION) \
	  --dtype $(VLLM_DTYPE) \
	  --max-model-len $(VLLM_MAX_LEN) \
	  --max-num-seqs $(VLLM_MAX_SEQS) \
	  --gpu-memory-utilization $(VLLM_GPU_UTIL) \
	  --seed $(SEED) \
	  --port $(VLLM_PORT)

labels:  ## Generate + grade labels. BENCH=<name>|all; LIMIT=N for a smoke run (first N items)
	$(RUN) python -m switchboard.labeling.generate --bench $(BENCH) $(if $(LIMIT),--limit $(LIMIT)) \
	  --vllm-gpu-util $(VLLM_GPU_UTIL) --vllm-max-len $(VLLM_MAX_LEN) --vllm-max-seqs $(VLLM_MAX_SEQS)

regrade:  ## Re-grade cached generations after a grader fix (no vLLM needed). BENCH=<name>|all
	$(RUN) python -m switchboard.labeling.generate --bench $(BENCH) --regrade

labels-summary:  ## Per-benchmark accuracy / base rate → results/labels-summary.json + markdown
	$(RUN) python -m switchboard.labeling.summary

# --- Phase 2: router v0 and the kill gate -------------------------------------------------------

train-v0:  ## Attempt 1: heuristic, random, v0 on GSM8K/MMLU/MBPP → R2 report (reproduction)
	$(RUN) python -m switchboard.router.evaluate --attempt 1

train-attempt2:  ## Attempt 2: wider mix, balanced weights, v0 + fine-tuned v1 x3 seeds (GPU) → R2
	$(RUN) python -m switchboard.router.evaluate --attempt 2

frontier-pilot:  ## Pilot: Claude answers 10 test prompts per benchmark; projects full cost (spends ≤ $2)
	$(RUN) python -m switchboard.labeling.frontier --pilot 10

frontier:  ## Claude answers test prompts (cached). MAX_USD=<ceiling> required; FRACTION=0.2 optional
	@test -n "$(MAX_USD)" || { echo "set MAX_USD, e.g. make frontier MAX_USD=25"; exit 1; }
	$(RUN) python -m switchboard.labeling.frontier --max-usd $(MAX_USD) $(if $(FRACTION),--fraction $(FRACTION))

sweep:  ## Threshold sweep → cost-quality curve, gate criteria 1-2, sensitivity → R2 + chart
	$(RUN) python -m switchboard.bench.sweep

readme:  ## Regenerate the README's numbers and the reliability diagram from results/
	$(RUN) python -m switchboard.bench.readme

review-sample:  ## Draw 30 random (prompt, generation, label) triples for the manual gate
	$(RUN) python -m switchboard.labeling.review --n 30 --seed $(SEED)
