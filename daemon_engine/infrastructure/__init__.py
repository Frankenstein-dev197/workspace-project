"""Infrastructure: Docker, deployment, and system management.

Integrates patterns from Vercel (deployment), Ansible (infrastructure as code),
and Firecracker (container/VM management). Provides the engine with
infrastructure management capabilities.
"""

from daemon_engine.infrastructure.docker_manager import DockerManager
from daemon_engine.infrastructure.deployment_manager import DeploymentManager

__all__ = ["DockerManager", "DeploymentManager"]
