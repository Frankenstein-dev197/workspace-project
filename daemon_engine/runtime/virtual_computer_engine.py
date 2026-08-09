"""Virtual Computer Engine: emulated computer environment for agents.

Inspired by Firecracker's microVM concept and DeepSeek-Reasonix's sandbox
abstraction. Provides a virtual filesystem, process table, and resource
simulation so agents can "operate a computer" in a controlled environment.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from daemon_engine.runtime.sandbox import Sandbox, SandboxConfig, ExecutionResult

logger = logging.getLogger(__name__)


@dataclass
class VirtualProcess:
    pid: int
    name: str
    command: str
    status: str = "running"  # running, completed, failed, killed
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result: ExecutionResult | None = None


class VirtualComputerEngine:
    """A virtual computer with filesystem, process management, and resource tracking."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.sandbox = Sandbox(config=config)
        self._processes: dict[int, VirtualProcess] = {}
        self._next_pid: int = 1000
        self._fs_tree: dict[str, Any] = {"root": {}}
        self._env_vars: dict[str, str] = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": str(self.sandbox.workdir),
            "USER": "daemon",
            "SHELL": "/bin/bash",
        }
        self._boot_time = time.time()
        logger.info("VirtualComputerEngine booted at %s, workdir=%s", self._boot_time, self.sandbox.workdir)

    @property
    def uptime(self) -> float:
        return time.time() - self._boot_time

    def execute(self, command: str, timeout: int | None = None) -> VirtualProcess:
        pid = self._next_pid
        self._next_pid += 1
        process = VirtualProcess(pid=pid, name=command.split()[0] if command else "unknown", command=command)
        self._processes[pid] = process
        result = self.sandbox.execute_shell(command, timeout=timeout)
        process.result = result
        process.status = "completed" if result.success else "failed"
        process.completed_at = time.time()
        return process

    def execute_code(self, code: str, timeout: int | None = None) -> VirtualProcess:
        pid = self._next_pid
        self._next_pid += 1
        process = VirtualProcess(pid=pid, name="python_script", command="python3 -c '...'")
        self._processes[pid] = process
        result = self.sandbox.execute_python(code, timeout=timeout)
        process.result = result
        process.status = "completed" if result.success else "failed"
        process.completed_at = time.time()
        return process

    def create_file(self, path: str, content: str) -> str:
        self.sandbox.write_file(path, content)
        self._update_fs_tree(path, "file")
        return f"Created file: {path} ({len(content)} bytes)"

    def read_file(self, path: str) -> str:
        return self.sandbox.read_file(path)

    def list_directory(self, path: str = ".") -> list[str]:
        files = self.sandbox.list_files()
        if path and path != ".":
            files = [f for f in files if f.startswith(path)]
        return files

    def get_process(self, pid: int) -> VirtualProcess | None:
        return self._processes.get(pid)

    def list_processes(self, status: str | None = None) -> list[VirtualProcess]:
        if status:
            return [p for p in self._processes.values() if p.status == status]
        return list(self._processes.values())

    def kill_process(self, pid: int) -> bool:
        process = self._processes.get(pid)
        if process and process.status == "running":
            process.status = "killed"
            process.completed_at = time.time()
            return True
        return False

    def set_env(self, key: str, value: str) -> None:
        self._env_vars[key] = value

    def get_env(self, key: str) -> str | None:
        return self._env_vars.get(key)

    def get_env_all(self) -> dict[str, str]:
        return dict(self._env_vars)

    def _update_fs_tree(self, path: str, node_type: str) -> None:
        parts = path.strip("/").split("/")
        current = self._fs_tree["root"]
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = {"type": node_type, "path": path}

    def system_info(self) -> dict[str, Any]:
        return {
            "uptime_seconds": self.uptime,
            "workdir": str(self.sandbox.workdir),
            "total_processes": len(self._processes),
            "running_processes": sum(1 for p in self._processes.values() if p.status == "running"),
            "completed_processes": sum(1 for p in self._processes.values() if p.status == "completed"),
            "failed_processes": sum(1 for p in self._processes.values() if p.status == "failed"),
            "files_in_fs": len(self.sandbox.list_files()),
            "env_vars": len(self._env_vars),
            "sandbox_info": self.sandbox.info(),
        }

    def shutdown(self) -> None:
        for pid, process in list(self._processes.items()):
            if process.status == "running":
                self.kill_process(pid)
        self.sandbox.cleanup()
        logger.info("VirtualComputerEngine shutdown after %.1fs uptime", self.uptime)
