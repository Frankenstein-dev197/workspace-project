"""DevOps Tools: infrastructure and deployment capabilities.

Integrates patterns from Ansible (playbook execution, inventory management)
and Firecracker (microVM management). Provides shell execution, Docker
operations, and infrastructure provisioning tools.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any

from daemon_engine.tools.tool_registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


class DevOpsTools:
    """DevOps and infrastructure management tools."""

    DANGEROUS_COMMANDS = [
        "rm -rf /", "sudo ", "shutdown", "reboot", "mkfs",
        "dd if=/dev/zero", ":(){:|:&};:", "fork bomb",
    ]

    def __init__(self, workdir: str | None = None) -> None:
        self.workdir = workdir or os.getcwd()
        self._allowed_dirs: list[str] = [self.workdir]

    def register_all(self, registry: ToolRegistry) -> None:
        registry.register(
            "bash",
            "Execute a shell command in the sandbox",
            self.execute_command,
            category="devops",
            parameters={"command": {"type": "string", "required": True}},
            is_safe=False,
        )
        registry.register(
            "file_read",
            "Read the contents of a file",
            self.read_file,
            category="devops",
            parameters={"path": {"type": "string", "required": True}},
        )
        registry.register(
            "file_write",
            "Write content to a file",
            self.write_file,
            category="devops",
            parameters={"path": {"type": "string", "required": True}, "content": {"type": "string", "required": True}},
            is_safe=False,
        )
        registry.register(
            "file_list",
            "List files in a directory",
            self.list_files,
            category="devops",
            parameters={"path": {"type": "string", "required": False}},
        )
        registry.register(
            "docker_build",
            "Build a Docker image from a Dockerfile",
            self.docker_build,
            category="devops",
            parameters={"dockerfile_path": {"type": "string", "required": True}, "tag": {"type": "string", "required": False}},
            is_safe=False,
        )
        registry.register(
            "docker_run",
            "Run a Docker container",
            self.docker_run,
            category="devops",
            parameters={"image": {"type": "string", "required": True}, "command": {"type": "string", "required": False}},
            is_safe=False,
        )
        registry.register(
            "ansible_playbook",
            "Execute an Ansible playbook (simulated)",
            self.ansible_playbook,
            category="devops",
            parameters={"playbook": {"type": "string", "required": True}, "inventory": {"type": "string", "required": False}},
            is_safe=False,
        )

    def execute_command(self, command: str, **kwargs: Any) -> ToolResult:
        for dangerous in self.DANGEROUS_COMMANDS:
            if dangerous in command:
                return ToolResult(
                    tool_name="bash",
                    success=False,
                    error=f"Blocked dangerous command: contains '{dangerous}'",
                )
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=kwargs.get("workdir", self.workdir),
                capture_output=True,
                text=True,
                timeout=kwargs.get("timeout", 120),
            )
            output = (result.stdout + result.stderr).strip()
            return ToolResult(
                tool_name="bash",
                success=result.returncode == 0,
                output=output[:10000],
                error=result.stderr.strip() if result.returncode != 0 else "",
                data={"returncode": result.returncode, "command": command},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(tool_name="bash", success=False, error="Command timed out (120s)")
        except Exception as exc:
            return ToolResult(tool_name="bash", success=False, error=str(exc))

    def read_file(self, path: str, **kwargs: Any) -> ToolResult:
        try:
            full_path = os.path.join(self.workdir, path) if not os.path.isabs(path) else path
            with open(full_path, "r", encoding="utf-8") as f:
                content = f.read()
            return ToolResult(
                tool_name="file_read",
                success=True,
                output=content[:10000],
                data={"path": path, "size": len(content)},
            )
        except Exception as exc:
            return ToolResult(tool_name="file_read", success=False, error=str(exc))

    def write_file(self, path: str, content: str, **kwargs: Any) -> ToolResult:
        try:
            full_path = os.path.join(self.workdir, path) if not os.path.isabs(path) else path
            os.makedirs(os.path.dirname(full_path) if os.path.dirname(full_path) else ".", exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(
                tool_name="file_write",
                success=True,
                output=f"Written {len(content)} bytes to {path}",
                data={"path": path, "bytes": len(content)},
            )
        except Exception as exc:
            return ToolResult(tool_name="file_write", success=False, error=str(exc))

    def list_files(self, path: str = ".", **kwargs: Any) -> ToolResult:
        try:
            full_path = os.path.join(self.workdir, path) if not os.path.isabs(path) else path
            entries = sorted(os.listdir(full_path))
            return ToolResult(
                tool_name="file_list",
                success=True,
                output="\n".join(entries),
                data={"path": path, "count": len(entries)},
            )
        except Exception as exc:
            return ToolResult(tool_name="file_list", success=False, error=str(exc))

    def docker_build(self, dockerfile_path: str, tag: str = "daemon-build", **kwargs: Any) -> ToolResult:
        if not self._docker_available():
            return ToolResult(
                tool_name="docker_build",
                success=True,
                output=f"[Simulated] Would build Docker image '{tag}' from {dockerfile_path}",
                data={"simulated": True, "dockerfile": dockerfile_path, "tag": tag},
            )
        cmd = f"docker build -t {tag} -f {dockerfile_path} ."
        return self.execute_command(cmd)

    def docker_run(self, image: str, command: str = "", **kwargs: Any) -> ToolResult:
        if not self._docker_available():
            return ToolResult(
                tool_name="docker_run",
                success=True,
                output=f"[Simulated] Would run container from image '{image}' with command '{command}'",
                data={"simulated": True, "image": image, "command": command},
            )
        cmd = f"docker run --rm {image} {command}".strip()
        return self.execute_command(cmd)

    def ansible_playbook(self, playbook: str, inventory: str = "localhost", **kwargs: Any) -> ToolResult:
        try:
            playbook_path = playbook if os.path.isabs(playbook) else os.path.join(self.workdir, playbook)
            if os.path.exists(playbook_path):
                with open(playbook_path, "r") as f:
                    playbook_content = f.read()
                return ToolResult(
                    tool_name="ansible_playbook",
                    success=True,
                    output=f"Playbook loaded from {playbook}:\n{playbook_content[:2000]}",
                    data={"playbook": playbook, "inventory": inventory, "content_length": len(playbook_content)},
                )
            return ToolResult(
                tool_name="ansible_playbook",
                success=True,
                output=f"[Simulated] Would execute Ansible playbook '{playbook}' against inventory '{inventory}'",
                data={"simulated": True, "playbook": playbook, "inventory": inventory},
            )
        except Exception as exc:
            return ToolResult(tool_name="ansible_playbook", success=False, error=str(exc))

    def _docker_available(self) -> bool:
        try:
            result = subprocess.run("docker --version", shell=True, capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False
