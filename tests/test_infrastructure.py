"""Tests for infrastructure (Docker and Deployment)."""

import pytest

from daemon_engine.infrastructure.docker_manager import DockerManager, ContainerConfig
from daemon_engine.infrastructure.deployment_manager import (
    DeploymentManager,
    DeployConfig,
    DeployTarget,
    DeployStatus,
)


class TestDockerManager:
    def test_availability(self):
        manager = DockerManager()
        assert isinstance(manager.is_available, bool)

    def test_build_simulated(self):
        manager = DockerManager()
        if not manager.is_available:
            result = manager.build("Dockerfile", "test-image", context=".")
            assert result["success"] is True
            assert result.get("simulated") is True

    def test_run_simulated(self):
        manager = DockerManager()
        if not manager.is_available:
            config = ContainerConfig(image="nginx", name="test-container")
            result = manager.run(config)
            assert result["success"] is True
            assert result.get("simulated") is True

    def test_list_containers(self):
        manager = DockerManager()
        containers = manager.list_containers()
        assert isinstance(containers, list)

    def test_stats(self):
        manager = DockerManager()
        stats = manager.stats()
        assert "available" in stats
        assert "total_containers" in stats

    def test_container_config_to_run_args(self):
        config = ContainerConfig(
            image="nginx:latest",
            name="test",
            ports={"8080": "80"},
            environment={"ENV": "test"},
        )
        args = config.to_run_args()
        assert "run" in args
        assert "-d" in args
        assert "--name" in args
        assert "nginx:latest" in args


class TestDeployConfig:
    def test_detect_project_type(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        config = DeployConfig(project_path=str(tmp_path))
        assert config.detect_project_type() == "node"

    def test_detect_python(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("requests")
        config = DeployConfig(project_path=str(tmp_path))
        assert config.detect_project_type() == "python"

    def test_auto_config(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        config = DeployConfig(project_path=str(tmp_path))
        config.auto_config()
        assert config.name == tmp_path.name
        assert config.build_command == "npm run build"
        assert config.output_dir == "dist"


class TestDeploymentManager:
    def test_deploy_local(self, tmp_path):
        (tmp_path / "requirements.txt").write_text("pytest")
        manager = DeploymentManager()
        config = DeployConfig(project_path=str(tmp_path), target=DeployTarget.LOCAL)
        result = manager.deploy(config)
        assert result.status in (DeployStatus.SUCCESS, DeployStatus.FAILED)
        assert result.deploy_id.startswith("deploy-")

    def test_list_deployments(self, tmp_path):
        manager = DeploymentManager()
        config = DeployConfig(project_path=str(tmp_path), target=DeployTarget.LOCAL)
        manager.deploy(config)
        deployments = manager.list_deployments()
        assert len(deployments) >= 1

    def test_stats(self):
        manager = DeploymentManager()
        stats = manager.stats()
        assert "total_deployments" in stats
        assert "successful" in stats

    def test_get_deployment(self, tmp_path):
        manager = DeploymentManager()
        config = DeployConfig(project_path=str(tmp_path), target=DeployTarget.LOCAL)
        result = manager.deploy(config)
        retrieved = manager.get_deployment(result.deploy_id)
        assert retrieved is not None
        assert retrieved.deploy_id == result.deploy_id

    def test_deploy_vercel_simulated(self, tmp_path):
        (tmp_path / "package.json").write_text("{}")
        manager = DeploymentManager()
        config = DeployConfig(
            project_path=str(tmp_path),
            target=DeployTarget.VERCEL,
            name="test-app",
        )
        result = manager.deploy(config)
        assert result.status in (DeployStatus.SUCCESS, DeployStatus.FAILED)
