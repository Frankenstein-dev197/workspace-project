"""Watchdog: detects agent loops and triggers recovery.

Integrates AutoGPT's WatchdogComponent pattern: monitors agent actions
for repetitive commands and loops. When detected, switches to a smarter
model and re-thinks the action.

Loop detection strategies:
1. Exact repetition: same command + same arguments
2. Partial repetition: same command with similar arguments
3. No command: agent didn't specify a tool
4. Circular: A → B → A pattern
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_LOOP_THRESHOLD = 3
DEFAULT_WINDOW_SIZE = 10
DEFAULT_CIRCULAR_WINDOW = 6


class LoopType(str, Enum):
    """Types of detected loops."""
    EXACT_REPETITION = "exact_repetition"
    PARTIAL_REPETITION = "partial_repetition"
    NO_COMMAND = "no_command"
    CIRCULAR = "circular"
    STUCK = "stuck"


@dataclass
class ActionRecord:
    """Record of a single agent action."""
    tool_name: str | None
    tool_input: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    result: str = ""
    success: bool = True


@dataclass
class LoopDetection:
    """Result of loop detection."""
    is_loop: bool
    loop_type: LoopType | None = None
    reason: str = ""
    repeated_action: ActionRecord | None = None
    repetition_count: int = 0
    suggested_action: str = ""


class Watchdog:
    """Monitors agent actions for loops and triggers recovery.

    When a loop is detected, the watchdog can:
    - Switch to a smarter model (big_brain mode)
    - Rewind the action history
    - Inject a re-think prompt
    - Escalate to a different strategy
    """

    def __init__(
        self,
        loop_threshold: int = DEFAULT_LOOP_THRESHOLD,
        window_size: int = DEFAULT_WINDOW_SIZE,
        circular_window: int = DEFAULT_CIRCULAR_WINDOW,
        fast_model: str = "fast",
        smart_model: str = "smart",
    ) -> None:
        self._loop_threshold = loop_threshold
        self._window_size = window_size
        self._circular_window = circular_window
        self._fast_model = fast_model
        self._smart_model = smart_model
        self._action_history: deque[ActionRecord] = deque(maxlen=window_size)
        self._repetition_counts: dict[str, int] = {}
        self._big_brain_active = False
        self._revert_big_brain = False
        self._stats = {
            "total_actions": 0,
            "loops_detected": 0,
            "by_type": {lt.value: 0 for lt in LoopType},
            "big_brain_activations": 0,
            "rewinds": 0,
        }

    @property
    def is_big_brain(self) -> bool:
        """Whether the smart model is currently active."""
        return self._big_brain_active

    @property
    def current_model(self) -> str:
        """Get the current model name."""
        return self._smart_model if self._big_brain_active else self._fast_model

    def record_action(
        self,
        tool_name: str | None,
        tool_input: dict[str, Any] | None = None,
        result: str = "",
        success: bool = True,
    ) -> LoopDetection:
        """Record an action and check for loops."""
        self._stats["total_actions"] += 1
        record = ActionRecord(
            tool_name=tool_name,
            tool_input=tool_input or {},
            result=result,
            success=success,
        )
        detection = self._detect_loop(record)
        if detection.is_loop:
            self._stats["loops_detected"] += 1
            if detection.loop_type:
                self._stats["by_type"][detection.loop_type.value] += 1
            self._trigger_recovery(detection)
        self._action_history.append(record)
        if self._revert_big_brain and not detection.is_loop:
            self._big_brain_active = False
            self._revert_big_brain = False
        return detection

    def _detect_loop(self, current: ActionRecord) -> LoopDetection:
        """Detect if the current action forms a loop."""
        if current.tool_name is None:
            return LoopDetection(
                is_loop=True,
                loop_type=LoopType.NO_COMMAND,
                reason="Agent did not specify a command",
                repeated_action=current,
            )
        history = list(self._action_history)
        if not history:
            return LoopDetection(is_loop=False)
        key = self._action_key(current)
        self._repetition_counts[key] = self._repetition_counts.get(key, 0) + 1
        if self._repetition_counts[key] >= self._loop_threshold:
            return LoopDetection(
                is_loop=True,
                loop_type=LoopType.EXACT_REPETITION,
                reason=f"Repetitive command detected ({current.tool_name})",
                repeated_action=current,
                repetition_count=self._repetition_counts[key],
                suggested_action="Try a different approach or tool",
            )
        if len(history) >= 2:
            prev = history[-1]
            if prev.tool_name == current.tool_name and prev.tool_input == current.tool_input:
                consecutive = 1
                for i in range(len(history) - 2, -1, -1):
                    if (
                        history[i].tool_name == current.tool_name
                        and history[i].tool_input == current.tool_input
                    ):
                        consecutive += 1
                    else:
                        break
                if consecutive >= self._loop_threshold - 1:
                    return LoopDetection(
                        is_loop=True,
                        loop_type=LoopType.EXACT_REPETITION,
                        reason=f"Consecutive repetition ({current.tool_name} x{consecutive + 1})",
                        repeated_action=current,
                        repetition_count=consecutive + 1,
                    )
        if len(history) >= self._circular_window:
            recent = history[-(self._circular_window):] + [current]
            if self._detect_circular(recent):
                return LoopDetection(
                    is_loop=True,
                    loop_type=LoopType.CIRCULAR,
                    reason="Circular pattern detected (A → B → A)",
                    repeated_action=current,
                    suggested_action="Break the cycle with a different strategy",
                )
        if len(history) >= 3:
            recent = history[-3:]
            if all(r.tool_name == current.tool_name for r in recent) and not any(r.success for r in recent):
                return LoopDetection(
                    is_loop=True,
                    loop_type=LoopType.STUCK,
                    reason=f"Stuck on failing command ({current.tool_name})",
                    repeated_action=current,
                    suggested_action="Try a different tool or approach",
                )
        return LoopDetection(is_loop=False)

    def _action_key(self, record: ActionRecord) -> str:
        """Generate a hashable key for an action."""
        input_str = str(sorted(record.tool_input.items()))
        return f"{record.tool_name}:{input_str}"

    def _detect_circular(self, actions: list[ActionRecord]) -> bool:
        """Detect circular A → B → A pattern."""
        n = len(actions)
        for cycle_len in range(2, n // 2 + 1):
            pattern = actions[:cycle_len]
            is_circular = True
            for i in range(cycle_len, n):
                expected = pattern[i % cycle_len]
                actual = actions[i]
                if expected.tool_name != actual.tool_name:
                    is_circular = False
                    break
            if is_circular and n >= cycle_len * 2:
                return True
        return False

    def _trigger_recovery(self, detection: LoopDetection) -> None:
        """Trigger recovery actions for a detected loop."""
        if not self._big_brain_active:
            self._big_brain_active = True
            self._revert_big_brain = True
            self._stats["big_brain_activations"] += 1
            logger.info(
                "Watchdog: switching to SMART_LLM due to %s",
                detection.loop_type.value if detection.loop_type else "unknown",
            )

    def rewind(self, steps: int = 1) -> list[ActionRecord]:
        """Rewind the action history by N steps."""
        self._stats["rewinds"] += 1
        rewound: list[ActionRecord] = []
        for _ in range(min(steps, len(self._action_history))):
            rewound.append(self._action_history.pop())
        return rewound

    def clear_history(self) -> None:
        """Clear action history and repetition counts."""
        self._action_history.clear()
        self._repetition_counts.clear()
        self._big_brain_active = False
        self._revert_big_brain = False

    def get_history(self) -> list[ActionRecord]:
        """Get the action history."""
        return list(self._action_history)

    def get_recovery_prompt(self, detection: LoopDetection) -> str:
        """Generate a recovery prompt for the agent."""
        prompts = {
            LoopType.EXACT_REPETITION: (
                f"You've repeated '{detection.repeated_action.tool_name}' "
                f"{detection.repetition_count} times. Try a different approach."
            ),
            LoopType.NO_COMMAND: (
                "You didn't specify a command. Please choose a tool to use."
            ),
            LoopType.CIRCULAR: (
                "You're in a circular pattern (A → B → A). "
                "Break the cycle with a fundamentally different strategy."
            ),
            LoopType.STUCK: (
                f"You're stuck on '{detection.repeated_action.tool_name}'. "
                "The tool keeps failing. Try a different tool."
            ),
            LoopType.PARTIAL_REPETITION: (
                "You're repeating similar actions. Consider a new approach."
            ),
        }
        if detection.loop_type:
            return prompts.get(detection.loop_type, "Loop detected. Re-think your approach.")
        return "Loop detected. Re-think your approach."

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "big_brain_active": self._big_brain_active,
            "current_model": self.current_model,
            "history_size": len(self._action_history),
            "tracked_actions": len(self._repetition_counts),
        }
