import json

import pytest

from switchboard.config import Settings
from switchboard.labeling.graders.code import CodeGrader
from switchboard.labeling.sandbox import DockerSandbox


@pytest.fixture
def sandbox() -> DockerSandbox:
    return DockerSandbox.from_settings(Settings())


def _flag_value(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_command_has_no_network(sandbox: DockerSandbox) -> None:
    assert _flag_value(sandbox.command("x"), "--network") == "none"


def test_command_enforces_timeout(sandbox: DockerSandbox) -> None:
    cmd = sandbox.command("x")
    i = cmd.index("timeout")
    assert cmd[i + 1 : i + 4] == ["-s", "KILL", str(sandbox.timeout_s)]
    assert sandbox.timeout_s > 0


def test_command_caps_resources_and_privileges(sandbox: DockerSandbox) -> None:
    cmd = sandbox.command("x")
    assert _flag_value(cmd, "--memory") == _flag_value(cmd, "--memory-swap")
    assert "--read-only" in cmd
    assert _flag_value(cmd, "--cap-drop") == "ALL"
    assert _flag_value(cmd, "--user") != "0"
    assert "--pids-limit" in cmd
    for mount_flag in ("-v", "--volume", "--mount"):
        assert mount_flag not in cmd


def test_command_uses_digest_pinned_image(sandbox: DockerSandbox) -> None:
    assert "@sha256:" in sandbox.image
    assert sandbox.image in sandbox.command("x")


# --- Live checks: need a Docker daemon (make test-docker) ---------------------------------------


@pytest.mark.docker
def test_live_passing_program(sandbox: DockerSandbox) -> None:
    assert sandbox.run("assert 1 + 1 == 2\n").passed


@pytest.mark.docker
def test_live_failing_program(sandbox: DockerSandbox) -> None:
    assert not sandbox.run("assert 1 + 1 == 3\n").passed


@pytest.mark.docker
def test_live_network_is_unreachable(sandbox: DockerSandbox) -> None:
    program = (
        "import socket\nsocket.setdefaulttimeout(3)\nsocket.create_connection(('1.1.1.1', 53))\n"
    )
    assert not sandbox.run(program).passed


@pytest.mark.docker
def test_live_infinite_loop_times_out() -> None:
    fast = DockerSandbox(image=Settings().sandbox_image, timeout_s=2)
    result = fast.run("while True:\n    pass\n")
    assert not result.passed
    assert result.timed_out


@pytest.mark.docker
def test_live_host_filesystem_not_writable(sandbox: DockerSandbox) -> None:
    assert not sandbox.run("open('/usr/pwned', 'w').write('x')\n").passed


@pytest.mark.docker
def test_live_code_grader_end_to_end(sandbox: DockerSandbox) -> None:
    ref = json.dumps({"test_setup_code": "", "test_list": ["assert sq(3) == 9"]})
    grader = CodeGrader("mbpp", sandbox)
    assert grader("```python\ndef sq(x):\n    return x * x\n```", ref)
    assert not grader("```python\ndef sq(x):\n    return x + x\n```", ref)
