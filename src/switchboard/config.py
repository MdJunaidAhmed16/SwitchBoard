"""Single settings object for the whole project.

Everything configurable is read here, from environment variables prefixed ``SWITCHBOARD_`` (or a
local ``.env``). No other module reads ``os.environ``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SWITCHBOARD_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Local model -------------------------------------------------------------------------
    # Every label depends on this checkpoint. Changing either field invalidates data/labels/.
    local_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    local_model_revision: str = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    local_base_url: str = "http://localhost:8001/v1"

    # --- Frontier model (Claude Opus 5.5 via OpenRouter) -----------------------------------------
    # The cost table reads these, so it cannot drift from what was actually called. Prices are
    # OpenRouter's listed rates for this model in USD per 1M tokens (checked 2026-10-03).
    frontier_base_url: str = "https://openrouter.ai/api/v1"
    frontier_model_id: str = "anthropic/claude-opus-5.5"
    frontier_price_in_per_mtok: float = 4.00
    frontier_price_out_per_mtok: float = 20.00
    # Hard ceiling on what one frontier run may spend (09-security, cost guardrail 4). A run that
    # would exceed it stops, keeps what it has, and writes partial results. Override per run.
    frontier_max_usd_per_run: float = Field(default=2.0, gt=0)
    frontier_concurrency: int = Field(default=8, ge=1)
    # Read from OPENROUTER_API_KEY (environment or the gitignored .env). Never logged or printed.
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "SWITCHBOARD_OPENROUTER_API_KEY"),
    )

    # --- Router ------------------------------------------------------------------------------
    # One backbone for v0 (frozen) and v1 (fine-tuned), so v1's gain over v0 is the value of the
    # fine-tuning alone. Pinned by commit hash like the local model.
    router_encoder_id: str = "BAAI/bge-base-en-v1.5"
    router_encoder_revision: str = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    router_max_tokens: int = Field(default=512, ge=1)
    seed: int = 0

    # --- Label generation --------------------------------------------------------------------
    label_concurrency: int = Field(default=32, ge=1)
    request_timeout_s: float = Field(default=300.0, gt=0)
    max_retries: int = Field(default=4, ge=0)

    # --- Code-execution sandbox (see 00-local-setup, "Code execution safety") ----------------
    # Pinned by digest, not tag: a base image that changes under us changes code-grader results.
    sandbox_image: str = (
        "python:3.11-slim@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e"
    )
    sandbox_timeout_s: int = Field(default=10, ge=1)
    sandbox_memory: str = "512m"
    sandbox_cpus: str = "1"
    sandbox_workers: int = Field(default=4, ge=1)

    # --- Paths -------------------------------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    results_dir: Path = REPO_ROOT / "results"
    reports_dir: Path = REPO_ROOT / "reports"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cache_path(self) -> Path:
        return self.data_dir / "cache" / "generations.sqlite"

    @property
    def labels_dir(self) -> Path:
        return self.data_dir / "labels"

    @property
    def splits_path(self) -> Path:
        return self.data_dir / "splits.yaml"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
