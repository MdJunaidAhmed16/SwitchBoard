import pytest

from switchboard.config import Settings


def test_env_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SWITCHBOARD_LABEL_CONCURRENCY", "7")
    assert Settings().label_concurrency == 7


def test_model_revision_is_a_commit_hash_not_a_branch() -> None:
    # 09-security: checkpoints are referenced by revision hash, never by branch name.
    revision = Settings().local_model_revision
    assert len(revision) == 40
    int(revision, 16)


def test_sandbox_image_is_pinned_by_digest() -> None:
    assert "@sha256:" in Settings().sandbox_image


def test_openrouter_key_is_read_and_never_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-not-a-real-key")
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.openrouter_api_key is not None
    assert settings.openrouter_api_key.get_secret_value() == "sk-or-test-not-a-real-key"
    assert "sk-or-test" not in repr(settings)
    assert "sk-or-test" not in str(settings.model_dump())


def test_missing_openrouter_key_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("SWITCHBOARD_OPENROUTER_API_KEY", raising=False)
    assert Settings(_env_file=None).openrouter_api_key is None  # type: ignore[call-arg]
