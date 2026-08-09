"""Docker manager: container lifecycle management.

Integrates Docker container operations with the engine, inspired by
the Docker CLI patterns and Ansible's container modules.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    image: str
    name: str = ""
    command: str = ""
    ports: dict[str, str] = field(default_factory=dict)  # host:container
    volumes: dict[str, str] = field(default_factory=dict)  # host:container
    environment: dict[str, str] = field(default_factory=dict)
    network: str = ""
    detach: bool = True
    auto_remove: bool = False
    working_dir: str = ""

    def to_run_args(self) -> list[str]:
        args: list[str] = ["run"]
        if self.detach:
            args.append("-d")
        if self.auto_remove:
            args.append("--rm")
        if self.name:
            args.extend(["--name", self.name])
        for host_port, container_port in self.ports.items():
            args.extend(["-p", f"{host_port}:{container_port}"])
        for host_vol, container_vol in self.volumes.items():
            args.extend(["-v", f"{host_vol}:{container_vol}"])
        for key, value in self.environment.items():
            args.extend(["-e", f"{key}={value}"])
        if self.network:
            args.extend(["--network", self.network])
        if self.working_dir:
            args.extend(["-w", self.working_dir])
        args.append(self.image)
        if self.command:
            args.extend(self.command.split())
        return args


@dataclass
class ContainerInfo:
    id: str
    name: str
    image: str
    status: str
    ports: str = ""
    created: str = ""


class DockerManager:
    """Docker container lifecycle management."""

    def __init__(self) -> None:
        self._available = self._check_docker()

    def _check_docker(self) -> bool:
        try:
            result = subprocess.run(
                "docker --version", shell=True, capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

    @property
    def is_available(self) -> bool:
        return self._available

    def _run_docker(self, args: list[str], timeout: int = 120) -> tuple[bool, str, str]:
        cmd = ["docker"] + args
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as exc:
            return False, "", str(exc)

    def build(self, dockerfile_path: str, tag: str, context: str = ".", timeout: int = 300) -> dict[str, Any]:
        if not self._available:
            return {"success": True, "simulated": True, "tag": tag, "message": f"[Simulated] Would build {tag}"}
        success, stdout, stderr = self._run_docker(
            ["build", "-t", tag, "-f", dockerfile_path, context], timeout=timeout
        )
        return {
            "success": success,
            "tag": tag,
            "output": stdout[:5000],
            "error": stderr[:2000] if not success else "",
        }

    def run(self, config: ContainerConfig, timeout: int = 60) -> dict[str, Any]:
        if not self._available:
            return {
                "success": True,
                "simulated": True,
                "name": config.name,
                "image": config.image,
                "message": f"[Simulated] Would run {config.image}",
            }
        success, stdout, stderr = self._run_docker(config.to_run_args(), timeout=timeout)
        container_id = stdout.strip() if success else ""
        return {
            "success": success,
            "container_id": container_id,
            "name": config.name,
            "image": config.image,
            "error": stderr[:2000] if not success else "",
        }

    def list_containers(self, all_containers: bool = False) -> list[ContainerInfo]:
        if not self._available:
            return []
        args = ["ps", "--format", "{{json .}}"]
        if all_containers:
            args.append("-a")
        success, stdout, _ = self._run_docker(args)
        if not success:
            return []
        containers: list[ContainerInfo] = []
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            try:
                data = json.loads(line)
                containers.append(ContainerInfo(
                    id=data.get("ID", "")[:12],
                    name=data.get("Names", ""),
                    image=data.get("Image", ""),
                    status=data.get("Status", ""),
                    ports=data.get("Ports", ""),
                    created=data.get("RunningFor", ""),
                ))
            except json.JSONDecodeError:
                continue
        return containers

    def stop(self, container_id_or_name: str, timeout: int = 10) -> bool:
        if not self._available:
            return True
        success, _, _ = self._run_docker(["stop", "-t", str(timeout), container_id_or_name])
        return success

    def remove(self, container_id_or_name: str, force: bool = False) -> bool:
        if not self._available:
            return True
        args = ["rm"]
        if force:
            args.append("-f")
        args.append(container_id_or_name)
        success, _, _ = self._run_docker(args)
        return success

    def logs(self, container_id_or_name: str, tail: int = 100) -> str:
        if not self._available:
            return f"[Simulated] logs for {container_id_or_name}"
        success, stdout, _ = self._run_docker(["logs", "--tail", str(tail), container_id_or_name])
        return stdout[:10000] if success else ""

    def exec_in_container(self, container_id_or_name: str, command: str, timeout: int = 30) -> dict[str, Any]:
        if not self._available:
            return {"success": True, "simulated": True, "output": f"[Simulated] exec {command}"}
        success, stdout, stderr = self._run_docker(
            ["exec", container_id_or_name] + command.split(), timeout=timeout
        )
        return {"success": success, "output": stdout[:5000], "error": stderr[:2000] if not success else ""}

    def pull(self, image: str, timeout: int = 120) -> bool:
        if not self._available:
            return True
        success, _, _ = self._run_docker(["pull", image], timeout=timeout)
        return success

    def images(self) -> list[dict[str, str]]:
        if not self._available:
            return []
        success, stdout, _ = self._run_docker(["images", "--format", "{{json .}}"])
        if not success:
            return []
        images: list[dict[str, str]] = []
        for line in stdout.strip().split("\n"):
            if line:
                try:
                    images.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return images

    def compose_up(self, compose_file: str, timeout: int = 180) -> dict[str, Any]:
        if not self._available:
            return {"success": True, "simulated": True, "message": "[Simulated] docker-compose up"}
        success, stdout, stderr = self._run_docker(
            ["compose", "-f", compose_file, "up", "-d"], timeout=timeout
        )
        return {"success": success, "output": stdout[:5000], "error": stderr[:2000] if not success else ""}

    def compose_down(self, compose_file: str, timeout: int = 60) -> dict[str, Any]:
        if not self._available:
            return {"success": True, "simulated": True}
        success, stdout, stderr = self._run_docker(
            ["compose", "-f", compose_file, "down"], timeout=timeout
        )
        return {"success": success, "output": stdout[:5000], "error": stderr[:2000] if not success else ""}

    def stats(self) -> dict[str, Any]:
        containers = self.list_containers(all_containers=True)
        return {
            "available": self._available,
            "total_containers": len(containers),
            "running": sum(1 for c in containers if "Up" in c.status),
            "stopped": sum(1 for c in containers if "Exited" in c.status),
            "images": len(self.images()) if self._available else 0,
        }
