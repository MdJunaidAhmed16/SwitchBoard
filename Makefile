# Switchboard — every sanctioned entry point lives here.
# Runs on Linux / WSL2. See the phase targets below for what each stage needs.

SHELL := /bin/bash
.DEFAULT_GOAL := help

UV       ?= uv
RUN      := $(UV) run
BENCH    ?= all
SEED     ?= 0

# vLLM runs as an external server in its own tool environment, so its pinned torch/CUDA never
# collides with the project's training dependencies.
VLLM_VERSION   ?= 0.30.0
VLLM_PORT      ?= 8001
VLLM_GPU_UTIL  ?= 0.85
VLLM_MAX_LEN   ?= 4096
VLLM_MAX_SEQS  ?= 32
LOCAL_MODEL    ?= $(shell $(RUN) python -c "from switchboard.config import get_settings as g; print(g().local_model_id)")
LOCAL_REVISION ?= $(shell $(RUN) python -c "from switchboard.config import get_settings as g; print(g().local_model_revision)")

.PHONY: help install lint format typecheck test test-docker test-integration check \
        sandbox-pull vllm labels labels-summary review-sample

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

vllm:  ## Serve the local model with vLLM on the GPU (foreground)
	$(UV) tool run --python 3.11 --from 'vllm==$(VLLM_VERSION)' vllm serve $(LOCAL_MODEL) \
	  --revision $(LOCAL_REVISION) \
	  --dtype bfloat16 \
	  --max-model-len $(VLLM_MAX_LEN) \
	  --max-num-seqs $(VLLM_MAX_SEQS) \
	  --gpu-memory-utilization $(VLLM_GPU_UTIL) \
	  --seed $(SEED) \
	  --port $(VLLM_PORT)

labels:  ## Generate + grade labels. BENCH=gsm8k|math|mmlu|arc_challenge|bbh|humaneval|mbpp|all
	$(RUN) python -m switchboard.labeling.generate --bench $(BENCH) \
	  --vllm-gpu-util $(VLLM_GPU_UTIL) --vllm-max-len $(VLLM_MAX_LEN) --vllm-max-seqs $(VLLM_MAX_SEQS)

labels-summary:  ## Per-benchmark accuracy / base rate → results/labels-summary.json + markdown
	$(RUN) python -m switchboard.labeling.summary

review-sample:  ## Draw 30 random (prompt, generation, label) triples for the manual gate
	$(RUN) python -m switchboard.labeling.review --n 30 --seed $(SEED)
