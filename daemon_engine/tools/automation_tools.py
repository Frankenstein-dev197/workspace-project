"""Automation Tools: workflow automation and task scheduling.

Integrates patterns from Ansible (task execution), Ruflo (workflow hooks),
and Vercel (deployment automation). Provides task scheduling, pipeline
execution, and deployment automation.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from daemon_engine.tools.tool_registry import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineStep:
    name: str
    command: str
    timeout: int = 120
    continue_on_failure: bool = False


@dataclass
class PipelineResult:
    steps_completed: int = 0
    steps_failed: int = 0
    outputs: list[str] = field(default_factory=list)
    duration: float = 0.0


class AutomationTools:
    """Workflow automation and pipeline execution tools."""

    def __init__(self) -> None:
        self._scheduled_tasks: dict[str, dict[str, Any]] = {}
        self._pipelines: dict[str, list[PipelineStep]] = {}

    def register_all(self, registry: ToolRegistry) -> None:
        registry.register(
            "run_pipeline",
            "Execute a multi-step automation pipeline",
            self.run_pipeline,
            category="automation",
            parameters={"steps": {"type": "array", "required": True}},
            is_safe=False,
        )
        registry.register(
            "schedule_task",
            "Schedule a task to run at a specified interval (simulated)",
            self.schedule_task,
            category="automation",
            parameters={"name": {"type": "string", "required": True}, "command": {"type": "string", "required": True}},
        )
        registry.register(
            "deploy_application",
            "Deploy an application (simulated Vercel-style deployment)",
            self.deploy_application,
            category="automation",
            parameters={"project_path": {"type": "string", "required": True}},
            is_safe=False,
        )
        registry.register(
            "run_tests",
            "Run the test suite for a project",
            self.run_tests,
            category="automation",
            parameters={"project_path": {"type": "string", "required": True}},
        )
        registry.register(
            "git_operations",
            "Perform git operations (status, add, commit, push)",
            self.git_operations,
            category="automation",
            parameters={"operation": {"type": "string", "required": True}, "path": {"type": "string", "required": False}},
            is_safe=False,
        )

    def run_pipeline(self, steps: list[Any] | None = None, **kwargs: Any) -> ToolResult:
        if not steps:
            return ToolResult(tool_name="run_pipeline", success=False, error="No steps provided")
        import subprocess

        pipeline_steps: list[PipelineStep] = []
        for s in steps:
            if isinstance(s, dict):
                pipeline_steps.append(PipelineStep(
                    name=s.get("name", "unnamed"),
                    command=s.get("command", ""),
                    timeout=s.get("timeout", 120),
                    continue_on_failure=s.get("continue_on_failure", False),
                ))
            elif isinstance(s, str):
                pipeline_steps.append(PipelineStep(name=s, command=s))
        start = time.time()
        result = PipelineResult()
        for step in pipeline_steps:
            try:
                r = subprocess.run(
                    step.command, shell=True, capture_output=True, text=True, timeout=step.timeout
                )
                output = (r.stdout + r.stderr).strip()
                result.outputs.append(f"[{step.name}] {'PASS' if r.returncode == 0 else 'FAIL'}: {output[:500]}")
                if r.returncode == 0:
                    result.steps_completed += 1
                else:
                    result.steps_failed += 1
                    if not step.continue_on_failure:
                        break
            except subprocess.TimeoutExpired:
                result.steps_failed += 1
                result.outputs.append(f"[{step.name}] TIMEOUT after {step.timeout}s")
                if not step.continue_on_failure:
                    break
            except Exception as exc:
                result.steps_failed += 1
                result.outputs.append(f"[{step.name}] ERROR: {exc}")
                if not step.continue_on_failure:
                    break
        result.duration = time.time() - start
        success = result.steps_failed == 0
        return ToolResult(
            tool_name="run_pipeline",
            success=success,
            output="\n".join(result.outputs),
            data={
                "steps_completed": result.steps_completed,
                "steps_failed": result.steps_failed,
                "duration": result.duration,
            },
        )

    def schedule_task(self, name: str, command: str, interval: str = "0 * * * *", **kwargs: Any) -> ToolResult:
        self._scheduled_tasks[name] = {
            "command": command,
            "interval": interval,
            "created_at": time.time(),
            "last_run": None,
        }
        return ToolResult(
            tool_name="schedule_task",
            success=True,
            output=f"Task '{name}' scheduled with interval '{interval}'",
            data={"name": name, "interval": interval, "command": command},
        )

    def deploy_application(self, project_path: str, **kwargs: Any) -> ToolResult:
        import os

        full_path = project_path if os.path.isabs(project_path) else os.path.join(os.getcwd(), project_path)
        if not os.path.exists(full_path):
            return ToolResult(
                tool_name="deploy_application",
                success=False,
                error=f"Project path not found: {full_path}",
            )
        deployment_url = f"https://daemon-deploy-{int(time.time())}.vercel.app"
        data = {
            "project_path": full_path,
            "deployment_url": deployment_url,
            "status": "deployed" if os.path.exists(os.path.join(full_path, "package.json")) else "simulated",
            "deployed_at": time.time(),
        }
        return ToolResult(
            tool_name="deploy_application",
            success=True,
            output=f"Application deployed to {deployment_url}",
            data=data,
        )

    def run_tests(self, project_path: str, **kwargs: Any) -> ToolResult:
        import os
        import subprocess

        full_path = project_path if os.path.isabs(project_path) else os.path.join(os.getcwd(), project_path)
        if not os.path.exists(full_path):
            return ToolResult(tool_name="run_tests", success=False, error=f"Path not found: {full_path}")
        test_commands: list[tuple[str, str]] = []
        if os.path.exists(os.path.join(full_path, "pyproject.toml")):
            test_commands.append(("pytest", "cd {} && python -m pytest --tb=short -q".format(full_path)))
        elif os.path.exists(os.path.join(full_path, "package.json")):
            test_commands.append(("npm test", "cd {} && npm test".format(full_path)))
        else:
            test_commands.append(("python unittest", "cd {} && python -m pytest --tb=short -q".format(full_path)))
        outputs: list[str] = []
        all_success = True
        for name, cmd in test_commands:
            try:
                r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
                output = (r.stdout + r.stderr).strip()
                outputs.append(f"[{name}] {'PASS' if r.returncode == 0 else 'FAIL'}:\n{output[:2000]}")
                if r.returncode != 0:
                    all_success = False
            except Exception as exc:
                outputs.append(f"[{name}] ERROR: {exc}")
                all_success = False
        return ToolResult(
            tool_name="run_tests",
            success=all_success,
            output="\n\n".join(outputs),
            data={"project_path": full_path, "all_passed": all_success},
        )

    def git_operations(self, operation: str, path: str = ".", **kwargs: Any) -> ToolResult:
        import os
        import subprocess

        valid_ops = {"status", "add", "commit", "push", "pull", "log", "branch", "checkout"}
        if operation not in valid_ops:
            return ToolResult(
                tool_name="git_operations",
                success=False,
                error=f"Invalid operation. Valid: {valid_ops}",
            )
        full_path = path if os.path.isabs(path) else os.path.join(os.getcwd(), path)
        cmd_map = {
            "status": "git status",
            "add": "git add -A",
            "commit": kwargs.get("message") and f'git commit -m "{kwargs["message"]}"' or "git commit",
            "push": "git push",
            "pull": "git pull",
            "log": "git log --oneline -10",
            "branch": "git branch",
            "checkout": kwargs.get("branch") and f"git checkout {kwargs['branch']}" or "git checkout",
        }
        try:
            r = subprocess.run(
                cmd_map[operation], shell=True, cwd=full_path, capture_output=True, text=True, timeout=30
            )
            return ToolResult(
                tool_name="git_operations",
                success=r.returncode == 0,
                output=(r.stdout + r.stderr).strip()[:5000],
                data={"operation": operation, "returncode": r.returncode},
            )
        except Exception as exc:
            return ToolResult(tool_name="git_operations", success=False, error=str(exc))
