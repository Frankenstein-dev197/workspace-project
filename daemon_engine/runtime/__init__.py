"""Runtime engine: secure execution environments for agents.

Integrates concepts from Firecracker (microVM sandboxing) and DeepSeek-Reasonix
(internal/sandbox). Provides sandboxed execution, virtual computer emulation,
and resource isolation for agent-run code.
"""

from daemon_engine.runtime.sandbox import Sandbox, SandboxConfig, ExecutionResult
from daemon_engine.runtime.virtual_computer_engine import VirtualComputerEngine
from daemon_engine.runtime.cron_scheduler import CronScheduler, CronJob, cron_matches, validate_cron
from daemon_engine.runtime.worktree import WorktreeManager, WorktreeInfo, validate_worktree_name

__all__ = [
    "Sandbox",
    "SandboxConfig",
    "ExecutionResult",
    "VirtualComputerEngine",
    "CronScheduler",
    "CronJob",
    "cron_matches",
    "validate_cron",
    "WorktreeManager",
    "WorktreeInfo",
    "validate_worktree_name",
]
