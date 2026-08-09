"""Agent brain: step-level reasoning with planning and evaluation.

Integrates browser-use's AgentBrain pattern: each agent step includes
thinking, evaluation of previous goal, memory, and next goal. The plan
is a list of PlanItems with status tracking.

This provides structured reasoning at each step of the agent loop,
enabling self-reflection and plan adaptation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class PlanItemStatus(str, Enum):
    """Status of a plan item."""
    PENDING = "pending"
    CURRENT = "current"
    DONE = "done"
    SKIPPED = "skipped"


@dataclass
class PlanItem:
    """A single item in an agent's plan."""
    text: str
    status: PlanItemStatus = PlanItemStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    def mark_done(self) -> None:
        self.status = PlanItemStatus.DONE
        self.completed_at = time.time()

    def mark_current(self) -> None:
        self.status = PlanItemStatus.CURRENT

    def mark_skipped(self) -> None:
        self.status = PlanItemStatus.SKIPPED

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass
class AgentBrain:
    """Step-level reasoning structure (from browser-use AgentBrain).

    Each agent step produces:
    - thinking: internal reasoning
    - evaluation_previous_goal: did the last action work?
    - memory: what to remember
    - next_goal: what to do next
    """
    thinking: str | None = None
    evaluation_previous_goal: str | None = None
    memory: str | None = None
    next_goal: str | None = None
    current_plan_item: int | None = None
    plan_update: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "thinking": self.thinking,
            "evaluation_previous_goal": self.evaluation_previous_goal,
            "memory": self.memory,
            "next_goal": self.next_goal,
            "current_plan_item": self.current_plan_item,
            "plan_update": self.plan_update,
        }


@dataclass
class AgentStep:
    """A single step in the agent's execution history."""
    step_number: int
    brain: AgentBrain
    action: str = ""
    action_input: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    timestamp: float = field(default_factory=time.time)
    duration: float = 0.0
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_number": self.step_number,
            "brain": self.brain.to_dict(),
            "action": self.action,
            "action_input": self.action_input,
            "result": self.result,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "success": self.success,
            "error": self.error,
        }


class AgentPlanner:
    """Manages an agent's plan with step-level reasoning.

    Combines browser-use's plan management with the AgentBrain pattern.
    The plan is adaptive: items can be added, skipped, or reordered.
    """

    def __init__(self, initial_plan: list[str] | None = None) -> None:
        self._plan: list[PlanItem] = []
        self._current_index: int | None = None
        self._steps: list[AgentStep] = []
        self._step_counter = 0
        if initial_plan:
            for text in initial_plan:
                self.add_plan_item(text)

    def add_plan_item(self, text: str, index: int | None = None) -> PlanItem:
        """Add a plan item at the specified index (or append)."""
        item = PlanItem(text=text)
        if index is not None and 0 <= index <= len(self._plan):
            self._plan.insert(index, item)
        else:
            self._plan.append(item)
        if self._current_index is None and self._plan:
            self._current_index = 0
            self._plan[0].mark_current()
        return item

    def remove_plan_item(self, index: int) -> bool:
        """Remove a plan item by index."""
        if 0 <= index < len(self._plan):
            self._plan.pop(index)
            if self._current_index is not None:
                if index < self._current_index:
                    self._current_index -= 1
                elif index == self._current_index:
                    self._current_index = min(self._current_index, len(self._plan) - 1)
                    if self._current_index is not None and self._current_index >= 0:
                        self._plan[self._current_index].mark_current()
            return True
        return False

    def get_current_item(self) -> PlanItem | None:
        """Get the current plan item."""
        if self._current_index is not None and 0 <= self._current_index < len(self._plan):
            return self._plan[self._current_index]
        return None

    def advance(self) -> PlanItem | None:
        """Mark current item done and advance to next."""
        if self._current_index is not None and 0 <= self._current_index < len(self._plan):
            self._plan[self._current_index].mark_done()
            self._current_index += 1
            if self._current_index < len(self._plan):
                self._plan[self._current_index].mark_current()
                return self._plan[self._current_index]
            self._current_index = None
        return None

    def skip_current(self) -> PlanItem | None:
        """Skip the current item and advance."""
        if self._current_index is not None and 0 <= self._current_index < len(self._plan):
            self._plan[self._current_index].mark_skipped()
            self._current_index += 1
            if self._current_index < len(self._plan):
                self._plan[self._current_index].mark_current()
                return self._plan[self._current_index]
            self._current_index = None
        return None

    def update_plan(self, new_items: list[str]) -> None:
        """Replace the entire plan (preserving completed items)."""
        completed_texts = {item.text for item in self._plan if item.status == PlanItemStatus.DONE}
        self._plan = []
        self._current_index = None
        for text in new_items:
            item = PlanItem(text=text)
            if text in completed_texts:
                item.mark_done()
            self._plan.append(item)
        for i, item in enumerate(self._plan):
            if item.status != PlanItemStatus.DONE:
                item.mark_current()
                self._current_index = i
                break

    def record_step(
        self,
        brain: AgentBrain,
        action: str = "",
        action_input: dict[str, Any] | None = None,
        result: str = "",
        success: bool = True,
        error: str = "",
        duration: float = 0.0,
    ) -> AgentStep:
        """Record a step in the execution history."""
        self._step_counter += 1
        step = AgentStep(
            step_number=self._step_counter,
            brain=brain,
            action=action,
            action_input=action_input or {},
            result=result,
            success=success,
            error=error,
            duration=duration,
        )
        self._steps.append(step)
        return step

    def get_steps(self) -> list[AgentStep]:
        """Get all recorded steps."""
        return list(self._steps)

    def get_plan(self) -> list[PlanItem]:
        """Get all plan items."""
        return list(self._plan)

    def get_plan_summary(self) -> dict[str, Any]:
        """Get a summary of plan progress."""
        total = len(self._plan)
        done = sum(1 for i in self._plan if i.status == PlanItemStatus.DONE)
        skipped = sum(1 for i in self._plan if i.status == PlanItemStatus.SKIPPED)
        pending = sum(1 for i in self._plan if i.status == PlanItemStatus.PENDING)
        current = sum(1 for i in self._plan if i.status == PlanItemStatus.CURRENT)
        return {
            "total_items": total,
            "done": done,
            "skipped": skipped,
            "pending": pending,
            "current": current,
            "completion_rate": done / total if total > 0 else 0,
            "current_index": self._current_index,
            "steps_recorded": len(self._steps),
        }

    def is_complete(self) -> bool:
        """Check if all plan items are done or skipped."""
        return all(
            item.status in (PlanItemStatus.DONE, PlanItemStatus.SKIPPED)
            for item in self._plan
        ) if self._plan else False

    def reset(self) -> None:
        """Reset all plan items to pending."""
        for item in self._plan:
            item.status = PlanItemStatus.PENDING
            item.completed_at = None
        self._current_index = 0 if self._plan else None
        if self._plan:
            self._plan[0].mark_current()
        self._steps.clear()
        self._step_counter = 0

    def to_dict(self) -> dict[str, Any]:
        """Export full state."""
        return {
            "plan": [item.to_dict() for item in self._plan],
            "current_index": self._current_index,
            "steps": [step.to_dict() for step in self._steps],
            "step_counter": self._step_counter,
            "summary": self.get_plan_summary(),
        }
