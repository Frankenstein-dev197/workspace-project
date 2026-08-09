"""Task graph: persistent task management with dependencies.

Integrates learn-claude-code's s12_task_system pattern: file-persisted
task graph with blockedBy dependencies, claim/complete lifecycle, and
multi-agent ownership tracking.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task lifecycle states."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class GraphTask:
    """A task in the dependency graph."""
    id: str
    subject: str
    description: str = ""
    status: TaskStatus = TaskStatus.PENDING
    owner: str | None = None
    blocked_by: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    subtasks: list[str] = field(default_factory=list)
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphTask:
        if isinstance(data.get("status"), str):
            data["status"] = TaskStatus(data["status"])
        if "blocked_by" in data and "blockedBy" not in data:
            data["blocked_by"] = data.pop("blocked_by", [])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class TaskGraph:
    """Persistent task graph with dependency tracking.

    Tasks are stored as JSON files and support:
    - blockedBy dependencies (can_start checks)
    - claim/complete lifecycle with ownership
    - Dependency resolution (completing unblocks downstream)
    - Priority ordering
    - Subtask relationships
    """

    def __init__(self, tasks_dir: Path | str = ".tasks") -> None:
        self.tasks_dir = Path(tasks_dir)
        self.tasks_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, GraphTask] = {}
        self._load_all()

    def _task_path(self, task_id: str) -> Path:
        return self.tasks_dir / f"{task_id}.json"

    def _load_all(self) -> None:
        """Load all tasks from disk into cache."""
        for path in self.tasks_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                task = GraphTask.from_dict(data)
                self._cache[task.id] = task
            except Exception as exc:
                logger.warning("Failed to load task from %s: %s", path, exc)

    def create_task(
        self,
        subject: str,
        description: str = "",
        blocked_by: list[str] | None = None,
        depends_on: list[str] | None = None,
        priority: int = 0,
    ) -> GraphTask:
        """Create a new task."""
        task_id = f"task_{int(time.time())}_{uuid.uuid4().hex[:4]}"
        task = GraphTask(
            id=task_id,
            subject=subject,
            description=description,
            blocked_by=blocked_by or [],
            depends_on=depends_on or [],
            priority=priority,
        )
        self._save_task(task)
        self._cache[task.id] = task
        logger.info("Created task %s: %s", task_id, subject)
        return task

    def _save_task(self, task: GraphTask) -> None:
        """Save a task to disk."""
        self._task_path(task.id).write_text(
            json.dumps(task.to_dict(), indent=2, default=str)
        )

    def get_task(self, task_id: str) -> GraphTask | None:
        """Get a task by ID."""
        return self._cache.get(task_id)

    def list_tasks(
        self,
        status: TaskStatus | None = None,
        owner: str | None = None,
    ) -> list[GraphTask]:
        """List tasks, optionally filtered."""
        tasks = list(self._cache.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        if owner:
            tasks = [t for t in tasks if t.owner == owner]
        tasks.sort(key=lambda t: (-t.priority, t.created_at))
        return tasks

    def can_start(self, task_id: str) -> bool:
        """Check if all blockedBy dependencies are completed.

        Missing dependencies are treated as blocked.
        """
        task = self._cache.get(task_id)
        if not task:
            return False
        for dep_id in task.blocked_by:
            dep = self._cache.get(dep_id)
            if not dep or dep.status != TaskStatus.COMPLETED:
                return False
        return True

    def claim_task(self, task_id: str, owner: str) -> bool:
        """Claim a task: set owner and change status to in_progress."""
        task = self._cache.get(task_id)
        if not task or task.status != TaskStatus.PENDING:
            return False
        if not self.can_start(task_id):
            return False
        task.owner = owner
        task.status = TaskStatus.IN_PROGRESS
        task.updated_at = time.time()
        self._save_task(task)
        logger.info("Task %s claimed by %s", task_id, owner)
        return True

    def complete_task(self, task_id: str, result: str = "") -> bool:
        """Complete a task and report unblocked downstream tasks."""
        task = self._cache.get(task_id)
        if not task or task.status != TaskStatus.IN_PROGRESS:
            return False
        task.status = TaskStatus.COMPLETED
        task.result = result
        task.completed_at = time.time()
        task.updated_at = time.time()
        self._save_task(task)
        unblocked = self._find_unblocked(task_id)
        logger.info("Task %s completed. Unblocked: %s", task_id, unblocked)
        return True

    def fail_task(self, task_id: str, error: str = "") -> bool:
        """Mark a task as failed."""
        task = self._cache.get(task_id)
        if not task:
            return False
        task.status = TaskStatus.FAILED
        task.result = error
        task.updated_at = time.time()
        self._save_task(task)
        return True

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self._cache.get(task_id)
        if not task:
            return False
        task.status = TaskStatus.CANCELLED
        task.updated_at = time.time()
        self._save_task(task)
        return True

    def _find_unblocked(self, completed_id: str) -> list[str]:
        """Find tasks that became unblocked after completing a task."""
        unblocked: list[str] = []
        for task in self._cache.values():
            if completed_id in task.blocked_by and task.status == TaskStatus.PENDING:
                if self.can_start(task.id):
                    unblocked.append(task.id)
        return unblocked

    def get_ready_tasks(self) -> list[GraphTask]:
        """Get all pending tasks that can start (deps completed)."""
        return [
            t for t in self._cache.values()
            if t.status == TaskStatus.PENDING and self.can_start(t.id)
        ]

    def get_blocked_tasks(self) -> list[GraphTask]:
        """Get tasks that are blocked by incomplete dependencies."""
        return [
            t for t in self._cache.values()
            if t.status == TaskStatus.PENDING and not self.can_start(t.id)
        ]

    def add_subtask(self, parent_id: str, child_id: str) -> bool:
        """Add a subtask relationship."""
        parent = self._cache.get(parent_id)
        if not parent:
            return False
        if child_id not in parent.subtasks:
            parent.subtasks.append(child_id)
            parent.updated_at = time.time()
            self._save_task(parent)
        return True

    def get_dependencies(self, task_id: str) -> list[GraphTask]:
        """Get the direct dependencies of a task."""
        task = self._cache.get(task_id)
        if not task:
            return []
        return [self._cache[d] for d in task.blocked_by if d in self._cache]

    def get_dependents(self, task_id: str) -> list[GraphTask]:
        """Get tasks that depend on this task."""
        return [
            t for t in self._cache.values()
            if task_id in t.blocked_by
        ]

    def get_critical_path(self) -> list[str]:
        """Get the longest dependency chain (simplified)."""
        memo: dict[str, list[str]] = {}

        def visit(tid: str) -> list[str]:
            if tid in memo:
                return memo[tid]
            task = self._cache.get(tid)
            if not task:
                return []
            longest: list[str] = []
            for dep_id in task.blocked_by:
                dep_path = visit(dep_id)
                if len(dep_path) > len(longest):
                    longest = dep_path
            memo[tid] = longest + [tid]
            return memo[tid]

        all_paths = [visit(t.id) for t in self._cache.values()]
        return max(all_paths, key=len) if all_paths else []

    def stats(self) -> dict[str, Any]:
        total = len(self._cache)
        by_status = {s.value: 0 for s in TaskStatus}
        for task in self._cache.values():
            by_status[task.status.value] += 1
        return {
            "total_tasks": total,
            "by_status": by_status,
            "ready_tasks": len(self.get_ready_tasks()),
            "blocked_tasks": len(self.get_blocked_tasks()),
            "completion_rate": by_status["completed"] / total if total > 0 else 0,
        }

    def clear(self) -> None:
        """Remove all tasks from disk and cache."""
        for path in self.tasks_dir.glob("*.json"):
            path.unlink()
        self._cache.clear()

    def to_dict(self) -> list[dict[str, Any]]:
        """Export all tasks as dictionaries."""
        return [t.to_dict() for t in self._cache.values()]
