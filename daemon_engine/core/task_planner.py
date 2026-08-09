"""Task Planner: decomposes complex goals into executable tasks.

Integrates hierarchical task decomposition patterns from AutoGPT and Ruflo
with the todo-write/task-system patterns from learn-claude-code.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from daemon_engine.models.base import BaseLLM, get_default_llm

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"

    @property
    def is_terminal(self) -> bool:
        return self in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}

    @property
    def is_active(self) -> bool:
        return self in {TaskStatus.IN_PROGRESS, TaskStatus.BLOCKED}


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    parent_id: str | None = None
    subtask_ids: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.MEDIUM
    assigned_agent_id: str | None = None
    result: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: __import__("time").time())
    updated_at: float = field(default_factory=lambda: __import__("time").time())

    def add_subtask(self, subtask_id: str) -> None:
        if subtask_id not in self.subtask_ids:
            self.subtask_ids.append(subtask_id)

    def is_leaf(self) -> bool:
        return len(self.subtask_ids) == 0


class TaskPlanner:
    """Decomposes high-level goals into a tree of executable tasks."""

    DECOMPOSITION_PROMPT = (
        "You are a task planner. Break down the following goal into concrete, "
        "actionable subtasks. Return each subtask on a new line prefixed with '- '. "
        "Keep subtasks small and verifiable.\n\nGoal: {goal}"
    )

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self.llm = llm or get_default_llm()
        self._tasks: dict[str, Task] = {}

    def create_task(
        self,
        description: str,
        parent_id: str | None = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
    ) -> Task:
        task = Task(description=description, parent_id=parent_id, priority=priority)
        self._tasks[task.id] = task
        if parent_id and parent_id in self._tasks:
            self._tasks[parent_id].add_subtask(task.id)
        logger.info("Created task: %s", description[:80])
        return task

    def decompose(self, goal: str, parent_id: str | None = None, max_depth: int = 3) -> Task:
        root = self.create_task(description=goal, parent_id=parent_id)
        if max_depth <= 0:
            return root
        try:
            prompt = self.DECOMPOSITION_PROMPT.format(goal=goal)
            response = self.llm.chat([{"role": "user", "content": prompt}])
            subtasks = self._parse_subtasks(response)
            for sub_desc in subtasks:
                self.decompose(sub_desc, parent_id=root.id, max_depth=max_depth - 1)
        except Exception as exc:
            logger.warning("LLM decomposition failed, using heuristic: %s", exc)
            for sub in self._heuristic_decompose(goal):
                self.create_task(description=sub, parent_id=root.id)
        return root

    def _parse_subtasks(self, response: str) -> list[str]:
        tasks: list[str] = []
        for line in response.strip().split("\n"):
            line = line.strip()
            if line.startswith("- "):
                tasks.append(line[2:].strip())
            elif line and not line.startswith(("#", "Goal", "Subtask")):
                tasks.append(line)
        return tasks

    def _heuristic_decompose(self, goal: str) -> list[str]:
        return [
            f"Analyze requirements for: {goal}",
            f"Design approach for: {goal}",
            f"Implement solution for: {goal}",
            f"Test and verify: {goal}",
        ]

    def get_task(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def get_subtasks(self, task_id: str) -> list[Task]:
        task = self._tasks.get(task_id)
        if not task:
            return []
        return [self._tasks[sid] for sid in task.subtask_ids if sid in self._tasks]

    def get_ready_tasks(self) -> list[Task]:
        ready: list[Task] = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            if not task.is_leaf():
                children = self.get_subtasks(task.id)
                if not all(c.status.is_terminal for c in children):
                    continue
            ready.append(task)
        return ready

    def update_status(self, task_id: str, status: TaskStatus, result: str = "") -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        task.status = status
        task.result = result
        import time

        task.updated_at = time.time()
        return True

    def assign_agent(self, task_id: str, agent_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        task.assigned_agent_id = agent_id
        return True

    def get_task_tree(self, task_id: str | None = None, indent: int = 0) -> str:
        if task_id is None:
            roots = [t for t in self._tasks.values() if t.parent_id is None]
        else:
            roots = [self._tasks[task_id]] if task_id in self._tasks else []
        lines: list[str] = []
        for task in roots:
            prefix = "  " * indent
            status_icon = {
                TaskStatus.PENDING: "○",
                TaskStatus.IN_PROGRESS: "◐",
                TaskStatus.COMPLETED: "●",
                TaskStatus.FAILED: "✗",
                TaskStatus.CANCELLED: "⊘",
                TaskStatus.BLOCKED: "⚠",
            }.get(task.status, "?")
            lines.append(f"{prefix}{status_icon} {task.description[:60]}")
            for sub_id in task.subtask_ids:
                lines.append(self.get_task_tree(sub_id, indent + 1))
        return "\n".join(lines)

    def all_tasks(self) -> list[Task]:
        return list(self._tasks.values())
