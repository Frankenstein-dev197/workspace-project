"""Episodic action history: track agent actions and results with summaries.

Integrates AutoGPT EpisodicActionHistory pattern:
- Episode: a single action + result + optional summary
  - action: the proposed action (tool name, args, reasoning)
  - result: outcome (status, output, error, feedback)
  - summary: optional compressed summary for older episodes
  - format: human-readable step description
- EpisodicActionHistory: ordered list of episodes with cursor
  - register_action: start a new episode
  - register_result: complete the current episode
  - current_episode: the active episode
  - append_user_feedback: queue feedback for next prompt
  - to_messages: render history as conversation messages
    - full_message_count: latest N episodes in full
    - older episodes as summaries
    - max_tokens cap on total history
- ActionStatus: SUCCESS, ERROR, INTERRUPTED, DID_NOT_FINISH

The episodic structure pairs each action with its result, enabling
compression of older steps while keeping recent ones in full detail.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ActionStatus(Enum):
    """Status of an action result."""
    SUCCESS = "success"
    ERROR = "error"
    INTERRUPTED_BY_HUMAN = "interrupted_by_human"
    DID_NOT_FINISH = "did_not_finish"


@dataclass
class AgentAction:
    """A proposed agent action."""
    use_tool: str
    args: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    raw_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_tool": self.use_tool,
            "args": self.args,
            "reasoning": self.reasoning,
            "raw_message": self.raw_message,
        }


@dataclass
class ActionResult:
    """The result of executing an action."""
    status: ActionStatus = ActionStatus.SUCCESS
    output: str = ""
    reason: str = ""
    error: str = ""
    feedback: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "output": self.output,
            "reason": self.reason,
            "error": self.error,
            "feedback": self.feedback,
        }

    def __str__(self) -> str:
        if self.output:
            return self.output
        return f"[{self.status.value}]"


@dataclass
class Episode:
    """A single action + result + optional summary."""
    action: AgentAction
    result: ActionResult | None = None
    summary: str | None = None
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    @property
    def is_complete(self) -> bool:
        """Whether this episode has both action and result."""
        return self.result is not None

    def format(self) -> str:
        """Human-readable step description."""
        step = f"Executed `{self.action.use_tool}`\n"
        if self.action.reasoning:
            step += f'- **Reasoning:** "{self.action.reasoning}"\n'
        status = self.result.status.value if self.result else "did_not_finish"
        step += f"- **Status:** `{status}`\n"
        if self.result:
            if self.result.status == ActionStatus.SUCCESS:
                result_str = str(self.result)
                step += f"- **Output:** {result_str}"
            elif self.result.status == ActionStatus.ERROR:
                step += f"- **Reason:** {self.result.reason}\n"
                if self.result.error:
                    step += f"- **Error:** {self.result.error}\n"
            elif self.result.status == ActionStatus.INTERRUPTED_BY_HUMAN:
                step += f"- **Feedback:** {self.result.feedback}\n"
        return step

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "result": self.result.to_dict() if self.result else None,
            "summary": self.summary,
            "timestamp": self.timestamp,
        }


def _default_count_tokens(text: str) -> int:
    """Default token counter: ~4 chars per token."""
    return max(1, len(text) // 4)


class EpisodicActionHistory:
    """Utility container for an action history.

    Tracks episodes (action + result pairs) with a cursor pointing to
    the current (incomplete) episode. Supports compression of older
    episodes via summaries.
    """

    def __init__(self) -> None:
        self.episodes: list[Episode] = []
        self.cursor: int = 0
        self.pending_user_feedback: list[str] = []
        self._lock = threading.Lock()

    @property
    def current_episode(self) -> Episode | None:
        """The current (possibly incomplete) episode."""
        if self.cursor >= len(self.episodes):
            return None
        return self.episodes[self.cursor]

    def __len__(self) -> int:
        return len(self.episodes)

    def __bool__(self) -> bool:
        return len(self.episodes) > 0

    def __getitem__(self, key: int) -> Episode:
        return self.episodes[key]

    def register_action(self, action: AgentAction) -> None:
        """Register a new action, starting a new episode."""
        with self._lock:
            if self.current_episode is None or self.current_episode.is_complete:
                self.episodes.append(Episode(action=action, result=None))
                self.cursor = len(self.episodes) - 1
            elif self.current_episode.action:
                raise ValueError("Action for current cycle already set")

    def register_result(self, result: ActionResult) -> None:
        """Register the result for the current episode."""
        with self._lock:
            if not self.current_episode:
                raise RuntimeError(
                    "Cannot register result for cycle without action"
                )
            if self.current_episode.result:
                raise ValueError("Result for current cycle already set")
            self.current_episode.result = result
            self.cursor = len(self.episodes)

    def append_user_feedback(self, feedback: str) -> None:
        """Append user feedback to be included in the next prompt."""
        with self._lock:
            self.pending_user_feedback.append(feedback)

    def set_summary(self, index: int, summary: str) -> None:
        """Set the summary for an episode (for compression)."""
        with self._lock:
            if 0 <= index < len(self.episodes):
                self.episodes[index].summary = summary

    def get_messages(
        self,
        *,
        full_message_count: int = 4,
        max_tokens: int = 1024,
        count_tokens: Callable[[str], int] = _default_count_tokens,
    ) -> list[dict[str, str]]:
        """Render history as conversation messages.

        - Latest `full_message_count` episodes in full detail
        - Older episodes as summaries (or formatted if no summary)
        - Total tokens capped at `max_tokens`
        """
        with self._lock:
            episodes = list(self.episodes)

        if not episodes:
            return []

        messages: list[dict[str, str]] = []
        step_summaries: list[str] = []
        tokens = 0
        n_episodes = len(episodes)

        for i, episode in enumerate(reversed(episodes)):
            if i < full_message_count:
                # Full detail for recent episodes
                msg = episode.action.raw_message or episode.format()
                messages.insert(0, {"role": "assistant", "content": msg})
                tokens += count_tokens(msg)
                if episode.result:
                    result_msg = str(episode.result)
                    messages.insert(
                        1, {"role": "user", "content": f"Result: {result_msg}"}
                    )
                    tokens += count_tokens(result_msg)
                continue
            else:
                step_content = (
                    episode.summary if episode.summary else episode.format()
                )
                step = f"* Step {n_episodes - i}: {step_content}"
                step_tokens = count_tokens(step)
                if max_tokens and tokens + step_tokens > max_tokens:
                    break
                tokens += step_tokens
                step_summaries.insert(0, step)

        if step_summaries:
            summary_text = "\n\n".join(step_summaries)
            messages.insert(
                0,
                {
                    "role": "user",
                    "content": (
                        "## Progress on your Task so far\n"
                        "Here is a summary of the steps that you have executed "
                        "so far, use this as your consideration for determining "
                        "the next action!\n"
                        f"{summary_text}"
                    ),
                },
            )

        if self.pending_user_feedback:
            feedback = "\n".join(self.pending_user_feedback)
            messages.append(
                {"role": "user", "content": f"## User Feedback\n{feedback}"}
            )

        return messages

    def clear(self) -> None:
        """Clear all history."""
        with self._lock:
            self.episodes.clear()
            self.cursor = 0
            self.pending_user_feedback.clear()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict."""
        with self._lock:
            return {
                "episodes": [e.to_dict() for e in self.episodes],
                "cursor": self.cursor,
                "pending_user_feedback": list(self.pending_user_feedback),
            }

    @property
    def completed_count(self) -> int:
        """Number of completed episodes."""
        with self._lock:
            return sum(1 for e in self.episodes if e.is_complete)
