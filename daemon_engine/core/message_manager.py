"""Message manager: conversation history management with compaction.

Integrates browser-use's MessageManager pattern (message state tracking,
history compaction, token budgeting) for managing agent conversation
context efficiently.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_call_id: str = ""
    tokens_estimated: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "tokens_estimated": self.tokens_estimated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        return cls(
            role=MessageRole(data.get("role", "user")),
            content=data.get("content", ""),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            tool_calls=data.get("tool_calls", []),
            tool_call_id=data.get("tool_call_id", ""),
            tokens_estimated=data.get("tokens_estimated", 0),
        )


@dataclass
class CompactionSettings:
    """Settings for message compaction (from browser-use's MessageCompactionSettings)."""
    enabled: bool = True
    compact_every_n_steps: int = 25
    trigger_char_count: int | None = None
    trigger_token_count: int | None = None
    chars_per_token: float = 4.0
    keep_last_items: int = 6
    summary_max_chars: int = 6000

    def __post_init__(self) -> None:
        if self.trigger_char_count is None and self.trigger_token_count is not None:
            self.trigger_char_count = int(self.trigger_token_count * self.chars_per_token)
        elif self.trigger_char_count is None:
            self.trigger_char_count = 40000


class MessageManager:
    """Manages conversation message history with compaction support.

    Inspired by browser-use's MessageManager: tracks messages, estimates
    token usage, and compacts history when it exceeds the token budget.
    """

    def __init__(
        self,
        system_prompt: str = "",
        compaction: CompactionSettings | None = None,
    ) -> None:
        self._messages: list[Message] = []
        self._compaction = compaction or CompactionSettings()
        self._step_count: int = 0
        self._compaction_count: int = 0
        if system_prompt:
            self.add_message(MessageRole.SYSTEM, system_prompt)

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def total_chars(self) -> int:
        return sum(len(m.content) for m in self._messages)

    @property
    def estimated_tokens(self) -> int:
        return sum(m.tokens_estimated for m in self._messages)

    def add_message(
        self,
        role: MessageRole | str,
        content: str,
        metadata: dict[str, Any] | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
        tool_call_id: str = "",
    ) -> Message:
        if isinstance(role, str):
            role = MessageRole(role)
        tokens = self._estimate_tokens(content)
        msg = Message(
            role=role,
            content=content,
            metadata=metadata or {},
            tool_calls=tool_calls or [],
            tool_call_id=tool_call_id,
            tokens_estimated=tokens,
        )
        self._messages.append(msg)
        self._step_count += 1
        self._maybe_compact()
        return msg

    def add_user(self, content: str, **kwargs: Any) -> Message:
        return self.add_message(MessageRole.USER, content, **kwargs)

    def add_assistant(self, content: str, **kwargs: Any) -> Message:
        return self.add_message(MessageRole.ASSISTANT, content, **kwargs)

    def add_tool_result(self, tool_call_id: str, content: str, **kwargs: Any) -> Message:
        return self.add_message(
            MessageRole.TOOL, content, tool_call_id=tool_call_id, **kwargs
        )

    def get_messages(self, role: MessageRole | None = None) -> list[Message]:
        if role:
            return [m for m in self._messages if m.role == role]
        return list(self._messages)

    def get_recent(self, n: int = 5) -> list[Message]:
        return self._messages[-n:] if n > 0 else []

    def get_system_prompt(self) -> str:
        for msg in self._messages:
            if msg.role == MessageRole.SYSTEM:
                return msg.content
        return ""

    def to_message_list(self) -> list[dict[str, Any]]:
        return [m.to_dict() for m in self._messages]

    def clear(self, keep_system: bool = True) -> None:
        if keep_system:
            system_msgs = [m for m in self._messages if m.role == MessageRole.SYSTEM]
            self._messages = system_msgs
        else:
            self._messages.clear()
        self._step_count = 0

    def _maybe_compact(self) -> None:
        if not self._compaction.enabled:
            return
        if self._step_count % self._compaction.compact_every_n_steps != 0:
            return
        if self.total_chars < (self._compaction.trigger_char_count or 40000):
            return
        self._compact()

    def _compact(self) -> None:
        keep_count = self._compaction.keep_last_items
        if len(self._messages) <= keep_count + 1:
            return
        system_msgs = [m for m in self._messages if m.role == MessageRole.SYSTEM]
        to_compact = [m for m in self._messages if m.role != MessageRole.SYSTEM][:-keep_count]
        keep_recent = [m for m in self._messages if m.role != MessageRole.SYSTEM][-keep_count:]
        if not to_compact:
            return
        summary_parts: list[str] = []
        for msg in to_compact:
            content_preview = msg.content[:200]
            summary_parts.append(f"[{msg.role.value}] {content_preview}")
        summary = "\n".join(summary_parts)
        if len(summary) > self._compaction.summary_max_chars:
            summary = summary[: self._compaction.summary_max_chars] + "\n... [compacted]"
        compacted_msg = Message(
            role=MessageRole.SYSTEM,
            content=f"[Compacted History - {len(to_compact)} messages]\n{summary}",
            metadata={"compacted": True, "original_count": len(to_compact)},
            tokens_estimated=self._estimate_tokens(summary),
        )
        self._messages = system_msgs + [compacted_msg] + keep_recent
        self._compaction_count += 1
        logger.info(
            "Compacted %d messages into summary (step %d, compaction #%d)",
            len(to_compact), self._step_count, self._compaction_count,
        )

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / self._compaction.chars_per_token)

    def stats(self) -> dict[str, Any]:
        return {
            "message_count": self.message_count,
            "total_chars": self.total_chars,
            "estimated_tokens": self.estimated_tokens,
            "step_count": self._step_count,
            "compaction_count": self._compaction_count,
            "compaction_enabled": self._compaction.enabled,
            "by_role": {
                role.value: sum(1 for m in self._messages if m.role == role)
                for role in MessageRole
            },
        }

    def save_state(self) -> dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self._messages],
            "step_count": self._step_count,
            "compaction_count": self._compaction_count,
        }

    def load_state(self, state: dict[str, Any]) -> None:
        self._messages = [Message.from_dict(m) for m in state.get("messages", [])]
        self._step_count = state.get("step_count", 0)
        self._compaction_count = state.get("compaction_count", 0)
