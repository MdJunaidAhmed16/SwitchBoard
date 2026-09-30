"""Isolated execution of model-generated code for the HumanEval and MBPP graders.

Model output is untrusted code. It runs in a throwaway container: no network, capped memory,
CPU and process count, non-root, read-only root filesystem, and a hard timeout. The program is
piped over stdin, so no host path is mounted into the container.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from dataclasses import dataclass
from typing import Protocol

from switchboard.config import Settings


@dataclass(frozen=True)
class ExecResult:
    passed: bool
    timed_out: bool
    returncode: int | None
    stderr_tail: str


class Sandbox(Protocol):
    def run(self, program: str) -> ExecResult: ...


# `timeout -s KILL` exits 137 on expiry; plain `timeout` exits 124.
_TIMEOUT_CODES = {124, 137}
# Grace on top of the in-container timeout, to cover container start-up.
_STARTUP_GRACE_S = 30


class DockerSandbox:
    """The only sandbox allowed to grade labels."""

    def __init__(
        self, image: str, timeout_s: int = 10, memory: str = "512m", cpus: str = "1"
    ) -> None:
        self.image = image
        self.timeout_s = timeout_s
        self.memory = memory
        self.cpus = cpus

    @classmethod
    def from_settings(cls, settings: Settings) -> DockerSandbox:
        return cls(
            image=settings.sandbox_image,
            timeout_s=settings.sandbox_timeout_s,
            memory=settings.sandbox_memory,
            cpus=settings.sandbox_cpus,
        )

    def command(self, name: str) -> list[str]:
        return [
            "docker", "run", "--rm", "-i",
            "--name", name,
            "--network", "none",
            "--memory", self.memory,
            "--memory-swap", self.memory,
            "--cpus", self.cpus,
            "--pids-limit", "64",
            "--read-only",
            "--tmpfs", "/tmp:rw,size=64m",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", "65534:65534",
            self.image,
            "timeout", "-s", "KILL", str(self.timeout_s),
            "python", "-I", "-",
        ]  # fmt: skip

    def run(self, program: str) -> ExecResult:
        name = f"switchboard-sandbox-{uuid.uuid4().hex[:12]}"
        try:
            proc = subprocess.run(
                self.command(name),
                input=program,
                capture_output=True,
                text=True,
                timeout=self.timeout_s + _STARTUP_GRACE_S,
                check=False,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True, check=False)
            return ExecResult(passed=False, timed_out=True, returncode=None, stderr_tail="")
        timed_out = proc.returncode in _TIMEOUT_CODES
        return ExecResult(
            passed=proc.returncode == 0,
            timed_out=timed_out,
            returncode=proc.returncode,
            stderr_tail=proc.stderr[-2000:],
        )


class TrustedSubprocessSandbox:
    """Runs code directly on the host. ONLY for hand-written test fixtures, never model output.

    Label generation refuses to use this class; see ``labeling.generate``.
    """

    def __init__(self, timeout_s: int = 10) -> None:
        self.timeout_s = timeout_s

    def run(self, program: str) -> ExecResult:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-"],
                input=program,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(passed=False, timed_out=True, returncode=None, stderr_tail="")
        return ExecResult(
            passed=proc.returncode == 0,
            timed_out=False,
            returncode=proc.returncode,
            stderr_tail=proc.stderr[-2000:],
        )
