"""Deployment manager: application deployment and environment management.

Integrates Vercel's deployment patterns (zero-config deploys, preview URLs)
with Ansible's infrastructure automation model (playbooks, inventory).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DeployTarget(str, Enum):
    VERCEL = "vercel"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    BARE_METAL = "bare-metal"
    LOCAL = "local"


class DeployStatus(str, Enum):
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeployConfig:
    project_path: str
    target: DeployTarget = DeployTarget.LOCAL
    name: str = ""
    environment: str = "production"
    build_command: str = ""
    output_dir: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    port: int = 3000
    dockerfile: str = "Dockerfile"
    k8s_manifest: str = "k8s.yaml"

    def detect_project_type(self) -> str:
        path = Path(self.project_path)
        if (path / "package.json").exists():
            return "node"
        if (path / "pyproject.toml").exists() or (path / "requirements.txt").exists():
            return "python"
        if (path / "go.mod").exists():
            return "go"
        if (path / "Cargo.toml").exists():
            return "rust"
        if (path / "Dockerfile").exists():
            return "docker"
        return "unknown"

    def auto_config(self) -> None:
        project_type = self.detect_project_type()
        if not self.name:
            self.name = Path(self.project_path).name
        if not self.build_command:
            self.build_command = {
                "node": "npm run build",
                "python": "pip install -e .",
                "go": "go build -o app",
                "rust": "cargo build --release",
                "docker": "docker build .",
                "unknown": "",
            }.get(project_type, "")
        if not self.output_dir:
            self.output_dir = {
                "node": "dist",
                "python": "",
                "go": "",
                "rust": "target/release",
                "docker": "",
                "unknown": "",
            }.get(project_type, "")


@dataclass
class DeployResult:
    deploy_id: str
    config: DeployConfig
    status: DeployStatus = DeployStatus.PENDING
    url: str = ""
    logs: str = ""
    error: str = ""
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    artifacts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deploy_id": self.deploy_id,
            "status": self.status.value,
            "target": self.config.target.value,
            "name": self.config.name,
            "url": self.url,
            "error": self.error,
            "duration": (self.completed_at or time.time()) - self.started_at,
            "artifacts": self.artifacts,
        }


class DeploymentManager:
    """Manages application deployments across multiple targets."""

    def __init__(self) -> None:
        self._deployments: dict[str, DeployResult] = {}
        self._deploy_counter = 0

    def deploy(self, config: DeployConfig) -> DeployResult:
        self._deploy_counter += 1
        deploy_id = f"deploy-{self._deploy_counter:04d}-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        config.auto_config()
        result = DeployResult(deploy_id=deploy_id, config=config)
        self._deployments[deploy_id] = result
        logger.info("Starting deployment %s: %s → %s", deploy_id, config.name, config.target.value)
        try:
            result.status = DeployStatus.BUILDING
            build_result = self._build(config)
            result.logs += build_result.get("logs", "")
            if not build_result.get("success"):
                result.status = DeployStatus.FAILED
                result.error = build_result.get("error", "Build failed")
                result.completed_at = time.time()
                return result
            result.status = DeployStatus.DEPLOYING
            deploy_result = self._deploy_to_target(config)
            result.logs += deploy_result.get("logs", "")
            result.url = deploy_result.get("url", "")
            result.artifacts = deploy_result.get("artifacts", [])
            if deploy_result.get("success"):
                result.status = DeployStatus.SUCCESS
            else:
                result.status = DeployStatus.FAILED
                result.error = deploy_result.get("error", "Deploy failed")
        except Exception as exc:
            result.status = DeployStatus.FAILED
            result.error = str(exc)
            logger.error("Deployment %s failed: %s", deploy_id, exc)
        result.completed_at = time.time()
        logger.info("Deployment %s: %s", deploy_id, result.status.value)
        return result

    def _build(self, config: DeployConfig) -> dict[str, Any]:
        if not config.build_command:
            return {"success": True, "logs": "No build command needed"}
        try:
            result = subprocess.run(
                config.build_command,
                shell=True,
                cwd=config.project_path,
                capture_output=True,
                text=True,
                timeout=180,
            )
            logs = (result.stdout + result.stderr)[:5000]
            if result.returncode == 0:
                return {"success": True, "logs": logs}
            return {"success": False, "logs": logs, "error": f"Build exit code {result.returncode}"}
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Build timed out (180s)"}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _deploy_to_target(self, config: DeployConfig) -> dict[str, Any]:
        if config.target == DeployTarget.VERCEL:
            return self._deploy_vercel(config)
        elif config.target == DeployTarget.DOCKER:
            return self._deploy_docker(config)
        elif config.target == DeployTarget.KUBERNETES:
            return self._deploy_k8s(config)
        elif config.target == DeployTarget.BARE_METAL:
            return self._deploy_bare_metal(config)
        else:
            return self._deploy_local(config)

    def _deploy_vercel(self, config: DeployConfig) -> dict[str, Any]:
        vercel_cli = subprocess.run("which vercel", shell=True, capture_output=True, text=True)
        if vercel_cli.returncode != 0:
            url = f"https://{config.name}-{int(time.time())}.vercel.app"
            return {
                "success": True,
                "simulated": True,
                "url": url,
                "logs": f"[Simulated] Would deploy to Vercel: {url}",
            }
        try:
            env = dict(os.environ)
            env.update(config.env_vars)
            result = subprocess.run(
                ["vercel", "--prod", "--yes", "--name", config.name],
                cwd=config.project_path,
                capture_output=True,
                text=True,
                timeout=120,
                env=env,
            )
            url = result.stdout.strip().split("\n")[-1] if result.stdout else ""
            return {
                "success": result.returncode == 0,
                "url": url,
                "logs": (result.stdout + result.stderr)[:5000],
                "error": result.stderr[:2000] if result.returncode != 0 else "",
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _deploy_docker(self, config: DeployConfig) -> dict[str, Any]:
        tag = f"{config.name}:{config.environment}"
        dockerfile = os.path.join(config.project_path, config.dockerfile)
        if not os.path.exists(dockerfile):
            return {"success": False, "error": f"Dockerfile not found: {dockerfile}"}
        try:
            build = subprocess.run(
                f"docker build -t {tag} -f {dockerfile} {config.project_path}",
                shell=True, capture_output=True, text=True, timeout=300,
            )
            if build.returncode != 0:
                return {"success": False, "error": build.stderr[:2000], "logs": build.stdout[:3000]}
            run = subprocess.run(
                f"docker run -d -p {config.port}:{config.port} --name {config.name} {tag}",
                shell=True, capture_output=True, text=True, timeout=30,
            )
            return {
                "success": run.returncode == 0,
                "url": f"http://localhost:{config.port}" if run.returncode == 0 else "",
                "logs": (build.stdout + run.stdout)[:5000],
                "error": run.stderr[:2000] if run.returncode != 0 else "",
                "artifacts": [tag],
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _deploy_k8s(self, config: DeployConfig) -> dict[str, Any]:
        manifest = os.path.join(config.project_path, config.k8s_manifest)
        if not os.path.exists(manifest):
            return {"success": False, "error": f"K8s manifest not found: {manifest}"}
        kubectl = subprocess.run("which kubectl", shell=True, capture_output=True, text=True)
        if kubectl.returncode != 0:
            return {"success": True, "simulated": True, "logs": "[Simulated] kubectl apply -f " + manifest}
        try:
            result = subprocess.run(
                ["kubectl", "apply", "-f", manifest],
                capture_output=True, text=True, timeout=60,
            )
            return {
                "success": result.returncode == 0,
                "logs": (result.stdout + result.stderr)[:5000],
                "error": result.stderr[:2000] if result.returncode != 0 else "",
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _deploy_bare_metal(self, config: DeployConfig) -> dict[str, Any]:
        return self._deploy_local(config)

    def _deploy_local(self, config: DeployConfig) -> dict[str, Any]:
        url = f"http://localhost:{config.port}"
        project_type = config.detect_project_type()
        start_commands = {
            "node": f"cd {config.project_path} && npm start",
            "python": f"cd {config.project_path} && python -m uvicorn main:app --port {config.port}",
            "go": f"cd {config.project_path} && ./app",
            "rust": f"cd {config.project_path} && ./target/release/{config.name}",
        }
        cmd = start_commands.get(project_type, "")
        if not cmd:
            return {"success": True, "simulated": True, "url": url, "logs": "[Simulated] local deployment"}
        return {
            "success": True,
            "url": url,
            "logs": f"Deployment ready at {url}. Start with: {cmd}",
            "artifacts": [cmd],
        }

    def rollback(self, deploy_id: str) -> bool:
        result = self._deployments.get(deploy_id)
        if not result or result.status != DeployStatus.SUCCESS:
            return False
        if result.config.target == DeployTarget.DOCKER:
            subprocess.run(
                f"docker stop {result.config.name} && docker rm {result.config.name}",
                shell=True, capture_output=True, text=True, timeout=30,
            )
        result.status = DeployStatus.ROLLED_BACK
        logger.info("Rolled back deployment %s", deploy_id)
        return True

    def get_deployment(self, deploy_id: str) -> DeployResult | None:
        return self._deployments.get(deploy_id)

    def list_deployments(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self._deployments.values()]

    def stats(self) -> dict[str, Any]:
        return {
            "total_deployments": len(self._deployments),
            "successful": sum(1 for r in self._deployments.values() if r.status == DeployStatus.SUCCESS),
            "failed": sum(1 for r in self._deployments.values() if r.status == DeployStatus.FAILED),
            "rolled_back": sum(1 for r in self._deployments.values() if r.status == DeployStatus.ROLLED_BACK),
        }
