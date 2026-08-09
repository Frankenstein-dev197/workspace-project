"""Orchestrator: top-level multi-agent workflow coordination.

Integrates orchestration patterns from DeerFlow (subagent dispatch and event
streams), Ruflo (swarm self-organization), and learn-claude-code (team
protocols). Decomposes goals, assigns tasks to agents, and aggregates results.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from daemon_engine.core.agent_engine import AgentResult
from daemon_engine.core.decision_system import DecisionSystem
from daemon_engine.core.task_planner import Task, TaskPlanner, TaskStatus
from daemon_engine.multi_agent.agent_manager import AgentManager
from daemon_engine.multi_agent.communication_system import CommunicationSystem, Message, MessageType
from daemon_engine.models.base import BaseLLM, get_default_llm

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Workflow:
    id: str
    goal: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    tasks: list[Task] = field(default_factory=list)
    results: list[AgentResult] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    error: str | None = None

    @property
    def is_complete(self) -> bool:
        return self.status in {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.CANCELLED}

    @property
    def progress(self) -> float:
        if not self.tasks:
            return 0.0
        done = sum(1 for t in self.tasks if t.status.is_terminal)
        return done / len(self.tasks)


class Orchestrator:
    """Coordinates multi-agent workflows from goal to completion."""

    def __init__(
        self,
        llm: BaseLLM | None = None,
        agent_manager: AgentManager | None = None,
        task_planner: TaskPlanner | None = None,
        communication: CommunicationSystem | None = None,
        decision_system: DecisionSystem | None = None,
        tool_registry: Any | None = None,
        memory: Any | None = None,
    ) -> None:
        self.llm = llm or get_default_llm()
        self.communication = communication or CommunicationSystem()
        self.agent_manager = agent_manager or AgentManager(
            llm=self.llm, communication=self.communication, memory=memory, tool_registry=tool_registry
        )
        self.task_planner = task_planner or TaskPlanner(llm=self.llm)
        self.decision_system = decision_system or DecisionSystem(llm=self.llm)
        self._workflows: dict[str, Workflow] = {}

    def execute_goal(self, goal: str, max_tasks: int = 20) -> Workflow:
        import uuid

        workflow = Workflow(id=str(uuid.uuid4()), goal=goal)
        self._workflows[workflow.id] = workflow
        logger.info("Starting workflow %s for goal: %s", workflow.id, goal[:80])
        workflow.status = WorkflowStatus.RUNNING
        try:
            root_task = self.task_planner.decompose(goal, max_depth=2)
            workflow.tasks = self.task_planner.all_tasks()
            self._execute_tasks(workflow, max_tasks)
            self._aggregate_results(workflow)
        except Exception as exc:
            logger.exception("Workflow %s failed", workflow.id)
            workflow.status = WorkflowStatus.FAILED
            workflow.error = str(exc)
        finally:
            workflow.completed_at = time.time()
        return workflow

    def _execute_tasks(self, workflow: Workflow, max_tasks: int) -> None:
        executed = 0
        while executed < max_tasks:
            ready = self.task_planner.get_ready_tasks()
            if not ready:
                break
            for task in ready[:max_tasks - executed]:
                self._assign_and_run_task(task, workflow)
                executed += 1
                if executed >= max_tasks:
                    break

    def _assign_and_run_task(self, task: Task, workflow: Workflow) -> None:
        role = self._select_role_for_task(task)
        agent = self.agent_manager.spawn_agent(role=role)
        self.agent_manager.update_status(agent.id, "running")
        self._notify_assignment(agent.id, task)
        result = agent.run(task)
        self.agent_manager.update_status(agent.id, "idle")
        self.agent_manager.record_completion(agent.id, result.status.value == "completed")
        workflow.results.append(result)
        if result.status.value == "completed":
            self.task_planner.update_status(task.id, TaskStatus.COMPLETED, result.output)
        else:
            self.task_planner.update_status(task.id, TaskStatus.FAILED, result.error or "Unknown error")
        self._notify_result(agent.id, result)

    def _select_role_for_task(self, task: Task) -> str:
        desc_lower = task.description.lower()
        if any(w in desc_lower for w in ("search", "find", "research", "investigate", "look up")):
            return "researcher"
        if any(w in desc_lower for w in ("code", "implement", "write", "debug", "refactor")):
            return "coder"
        if any(w in desc_lower for w in ("analyze", "examine", "evaluate", "assess")):
            return "analyst"
        if any(w in desc_lower for w in ("deploy", "build", "provision", "configure", "infrastructure")):
            return "devops"
        return "worker"

    def _notify_assignment(self, agent_id: str, task: Task) -> None:
        self.communication.send(Message(
            sender_id="orchestrator",
            recipient_id=agent_id,
            msg_type=MessageType.TASK_ASSIGNMENT,
            content=f"Task assigned: {task.description}",
            metadata={"task_id": task.id},
        ))

    def _notify_result(self, agent_id: str, result: AgentResult) -> None:
        self.communication.send(Message(
            sender_id=agent_id,
            recipient_id="orchestrator",
            msg_type=MessageType.TASK_RESULT,
            content=f"Task result: {result.status.value}",
            metadata={"task_id": result.task_id, "output": result.output[:200]},
        ))

    def _aggregate_results(self, workflow: Workflow) -> None:
        all_completed = all(
            self.task_planner.get_task(t.id) and self.task_planner.get_task(t.id).status == TaskStatus.COMPLETED
            for t in workflow.tasks
            if t.is_leaf()
        )
        workflow.status = WorkflowStatus.COMPLETED if all_completed else WorkflowStatus.FAILED
        if not all_completed:
            failed = [
                t.description for t in workflow.tasks
                if t.is_leaf() and t.status == TaskStatus.FAILED
            ]
            workflow.error = f"Failed tasks: {failed}"

    def get_workflow(self, workflow_id: str) -> Workflow | None:
        return self._workflows.get(workflow_id)

    def list_workflows(self) -> list[Workflow]:
        return list(self._workflows.values())

    def cancel_workflow(self, workflow_id: str) -> bool:
        workflow = self._workflows.get(workflow_id)
        if workflow and not workflow.is_complete:
            workflow.status = WorkflowStatus.CANCELLED
            return True
        return False

    def get_workflow_summary(self, workflow_id: str) -> dict[str, Any]:
        workflow = self._workflows.get(workflow_id)
        if not workflow:
            return {}
        return {
            "id": workflow.id,
            "goal": workflow.goal,
            "status": workflow.status.value,
            "progress": f"{workflow.progress:.0%}",
            "total_tasks": len(workflow.tasks),
            "completed_tasks": sum(1 for t in workflow.tasks if t.status == TaskStatus.COMPLETED),
            "failed_tasks": sum(1 for t in workflow.tasks if t.status == TaskStatus.FAILED),
            "results_count": len(workflow.results),
            "error": workflow.error,
            "task_tree": self.task_planner.get_task_tree(),
        }
