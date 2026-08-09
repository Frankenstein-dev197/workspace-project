"""Background tasks: async task execution with lifecycle tracking.

Integrates learn-claude-code s13 background tasks pattern:
- BackgroundTask: id, command, status (running, completed, failed)
- BackgroundTaskManager: manages daemon thread execution
- Slow operation detection: heuristic-based identification
- Result collection: notifications for completed tasks
- Thread-safe: all operations protected by locks

This enables the agent to dispatch slow operations (builds, installs,
tests) to background threads and continue working, collecting results
when ready.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

SLOW_KEYWORDS = [
    "install", "build", "test", "deploy", "compile",
    "docker build", "pip install", "npm install",
    "cargo build", "pytest", "make", "migrate",
]


class TaskStatus(str, Enum):
    """Status of a background task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BackgroundTask:
    """A background task with lifecycle tracking."""
    id: str
    command: str
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    started_at: float = 0.0
    completed_at: float = 0.0
    created_at: float = field(default_factory=time.time)
    thread: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return self.status == TaskStatus.RUNNING

    @property
    def is_completed(self) -> bool:
        return self.status == TaskStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        return self.status == TaskStatus.FAILED

    @property
    def is_ready(self) -> bool:
        """Whether the task result is ready to collect."""
        return self.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    @property
    def duration(self) -> float:
        """Duration in seconds (0 if not started)."""
        if not self.started_at:
            return 0.0
        end = self.completed_at if self.completed_at else time.time()
        return end - self.started_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "command": self.command,
            "tool_name": self.tool_name,
            "status": self.status.value,
            "result": self.result[:500],
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration": self.duration,
        }


class BackgroundTaskManager:
    """Manages background task execution in daemon threads.

    Tasks are dispatched to daemon threads and tracked. The manager
    provides methods to check status, collect results, and cancel tasks.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._stats = {
            "total_dispatched": 0,
            "total_completed": 0,
            "total_failed": 0,
            "total_cancelled": 0,
        }

    def is_slow_operation(self, tool_name: str, tool_input: dict[str, Any]) -> bool:
        """Heuristic: commands likely to take > 30s."""
        if tool_name != "bash":
            return False
        cmd = str(tool_input.get("command", "")).lower()
        return any(kw in cmd for kw in SLOW_KEYWORDS)

    def should_run_background(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> bool:
        """Determine if a task should run in background.

        Explicit request takes priority; fallback to heuristic.
        """
        if tool_input.get("run_in_background"):
            return True
        return self.is_slow_operation(tool_name, tool_input)

    def dispatch(
        self,
        func: Callable[..., str],
        tool_name: str = "",
        tool_input: dict[str, Any] | None = None,
        command: str = "",
    ) -> str:
        """Dispatch a function to run in a background thread.

        Returns the background task ID.
        """
        with self._lock:
            self._counter += 1
            bg_id = f"bg_{self._counter:04d}"

        tool_input = tool_input or {}
        if not command:
            command = tool_input.get("command", tool_name or func.__name__)

        task = BackgroundTask(
            id=bg_id,
            command=str(command),
            tool_name=tool_name,
            tool_input=tool_input,
        )

        def worker() -> None:
            task.started_at = time.time()
            with self._lock:
                task.status = TaskStatus.RUNNING
            try:
                result = func(**tool_input) if tool_input else func()
                with self._lock:
                    task.result = str(result) if result else ""
                    task.status = TaskStatus.COMPLETED
                    task.completed_at = time.time()
                    self._stats["total_completed"] += 1
            except Exception as exc:
                with self._lock:
                    task.error = str(exc)
                    task.status = TaskStatus.FAILED
                    task.completed_at = time.time()
                    self._stats["total_failed"] += 1
                logger.error("Background task %s failed: %s", bg_id, exc)

        with self._lock:
            self._tasks[bg_id] = task
            self._stats["total_dispatched"] += 1

        thread = threading.Thread(target=worker, daemon=True)
        task.thread = thread
        thread.start()
        logger.info("Background task dispatched: %s '%s'", bg_id, command[:40])
        return bg_id

    def dispatch_tool(
        self,
        handler: Callable[..., str],
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> str:
        """Dispatch a tool handler to background."""
        return self.dispatch(
            func=handler,
            tool_name=tool_name,
            tool_input=tool_input,
            command=tool_input.get("command", tool_name),
        )

    def get_task(self, bg_id: str) -> BackgroundTask | None:
        """Get a task by ID."""
        with self._lock:
            return self._tasks.get(bg_id)

    def get_status(self, bg_id: str) -> TaskStatus | None:
        """Get the status of a task."""
        task = self.get_task(bg_id)
        return task.status if task else None

    def collect_results(self) -> list[dict[str, Any]]:
        """Collect completed task results and remove from tracking.

        Returns notifications for ready tasks (completed or failed).
        """
        with self._lock:
            ready_ids = [bid for bid, t in self._tasks.items() if t.is_ready]
        notifications = []
        for bg_id in ready_ids:
            with self._lock:
                task = self._tasks.pop(bg_id, None)
            if not task:
                continue
            summary = task.result[:200] if task.result else ""
            notifications.append({
                "task_id": bg_id,
                "status": task.status.value,
                "command": task.command,
                "result": summary,
                "error": task.error,
                "duration": task.duration,
            })
        return notifications

    def has_ready(self) -> bool:
        """Return whether any tasks have results ready to collect."""
        with self._lock:
            return any(t.is_ready for t in self._tasks.values())

    def has_running(self) -> bool:
        """Return whether any tasks are still running."""
        with self._lock:
            return any(t.is_running for t in self._tasks.values())

    def list_tasks(self) -> list[BackgroundTask]:
        """List all tracked tasks."""
        with self._lock:
            return list(self._tasks.values())

    def cancel(self, bg_id: str) -> bool:
        """Mark a task as cancelled (cannot stop daemon thread)."""
        with self._lock:
            task = self._tasks.get(bg_id)
            if not task or not task.is_running:
                return False
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            self._stats["total_cancelled"] += 1
        logger.info("Background task cancelled: %s", bg_id)
        return True

    def wait_for(self, bg_id: str, timeout: float = 30.0) -> BackgroundTask | None:
        """Wait for a task to complete (with timeout)."""
        task = self.get_task(bg_id)
        if not task or not task.thread:
            return None
        task.thread.join(timeout=timeout)
        return self.get_task(bg_id)

    def wait_all(self, timeout: float = 60.0) -> None:
        """Wait for all running tasks to complete."""
        with self._lock:
            threads = [t.thread for t in self._tasks.values() if t.thread and t.is_running]
        for thread in threads:
            thread.join(timeout=timeout)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "active": sum(1 for t in self._tasks.values() if t.is_running),
                "ready": sum(1 for t in self._tasks.values() if t.is_ready),
                "tracked": len(self._tasks),
            }

    def clear(self) -> None:
        """Clear all completed tasks."""
        with self._lock:
            self._tasks = {bid: t for bid, t in self._tasks.items() if t.is_running}
