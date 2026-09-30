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
