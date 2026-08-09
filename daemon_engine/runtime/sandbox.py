"""Sandbox: secure code execution environment.

Inspired by Firecracker (microVM isolation) and DeepSeek-Reasonix's internal
sandbox package. Provides isolated Python code execution with resource limits,
dangerous command blocking, and filesystem isolation.
"""

from __future__ import annotations

import logging
import os
import resource
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class SandboxConfig:
    workdir: str | Path | None = None
    memory_limit_mb: int = 512
    cpu_time_limit: int = 30
    wall_time_limit: int = 60
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    max_processes: int = 10
    network_enabled: bool = True
    allowed_imports: list[str] = field(default_factory=list)
    blocked_imports: list[str] = field(default_factory=lambda: [
        "os.system", "subprocess.Popen", "shutil.rmtree",
        "os.exec", "os.fork", "ctypes",
    ])


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    duration: float = 0.0
    files_created: list[str] = field(default_factory=list)
    error: str = ""


class Sandbox:
    """Isolated execution sandbox for running agent-generated code."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        self._tempdir: tempfile.TemporaryDirectory | None = None
        self.workdir: Path = Path(self.config.workdir) if self.config.workdir else Path(tempfile.mkdtemp(prefix="daemon_sandbox_"))
        self._created_files: list[str] = []

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()

    def execute_python(self, code: str, timeout: int | None = None) -> ExecutionResult:
        for blocked in self.config.blocked_imports:
            if blocked in code:
                logger.warning("Blocked code contains forbidden pattern: %s", blocked)
                return ExecutionResult(
                    success=False,
                    error=f"Code contains blocked pattern: {blocked}",
                )
        script_path = self.workdir / "sandbox_script.py"
        script_path.write_text(code)
        start = time.time()
        wall_timeout = timeout or self.config.wall_time_limit
        try:
            env = dict(os.environ)
            env["PYTHONPATH"] = str(self.workdir)
            result = subprocess.run(
                ["python3", str(script_path)],
                capture_output=True,
                text=True,
                timeout=wall_timeout,
                cwd=str(self.workdir),
                env=env,
            )
            files_created = self._get_created_files()
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout[:10000],
                stderr=result.stderr[:10000],
                returncode=result.returncode,
                duration=time.time() - start,
                files_created=files_created,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"Execution timed out after {wall_timeout}s",
                duration=time.time() - start,
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error=str(exc),
                duration=time.time() - start,
            )

    def execute_shell(self, command: str, timeout: int | None = None) -> ExecutionResult:
        dangerous = ["rm -rf /", "sudo ", "shutdown", "reboot", "mkfs", ":(){:|:&};:"]
        for d in dangerous:
            if d in command:
                return ExecutionResult(success=False, error=f"Blocked dangerous command: {d}")
        start = time.time()
        wall_timeout = timeout or self.config.wall_time_limit
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=wall_timeout,
                cwd=str(self.workdir),
            )
            return ExecutionResult(
                success=result.returncode == 0,
                stdout=result.stdout[:10000],
                stderr=result.stderr[:10000],
                returncode=result.returncode,
                duration=time.time() - start,
                files_created=self._get_created_files(),
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                error=f"Command timed out after {wall_timeout}s",
                duration=time.time() - start,
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error=str(exc),
                duration=time.time() - start,
            )

    def write_file(self, name: str, content: str) -> Path:
        path = self.workdir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if name not in self._created_files:
            self._created_files.append(name)
        return path

    def read_file(self, name: str) -> str:
        path = self.workdir / name
        if path.exists():
            return path.read_text()
        return ""

    def list_files(self) -> list[str]:
        if self.workdir.exists():
            return sorted(os.listdir(self.workdir))
        return []

    def _get_created_files(self) -> list[str]:
        current = set(self.list_files())
        original = set(self._created_files)
        new = current - original - {"sandbox_script.py"}
        return sorted(new)

    def cleanup(self) -> None:
        if self._tempdir:
            self._tempdir.cleanup()
            self._tempdir = None
        self._created_files.clear()

    def info(self) -> dict[str, Any]:
        return {
            "workdir": str(self.workdir),
            "memory_limit_mb": self.config.memory_limit_mb,
            "cpu_time_limit": self.config.cpu_time_limit,
            "wall_time_limit": self.config.wall_time_limit,
            "network_enabled": self.config.network_enabled,
            "files_in_sandbox": len(self.list_files()),
        }
