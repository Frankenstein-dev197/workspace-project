"""Conversation summary buffer: sliding-window memory with running summary.

Integrates LangChain ConversationSummaryBufferMemory pattern:
- ConversationSummaryBuffer: buffer with running summary under token limit
  - max_token_limit: cap on buffer tokens (older messages summarized)
  - moving_summary_buffer: running summary of pruned messages
  - save_context: add message pair, then prune
  - prune: when over limit, pop oldest messages and summarize them
  - load_memory_variables: return summary + recent messages
- Message role enum: HUMAN, AI, SYSTEM
- ConversationMessage: single message with role and content
- estimate_tokens: simple token estimation (char-based fallback)

Provides a running summary of the conversation together with the most
recent messages under the constraint that the total number of tokens
does not exceed a certain limit. When the limit is exceeded, the oldest
messages are removed and folded into the running summary.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Role of a conversation message."""
    HUMAN = "human"
    AI = "ai"
    SYSTEM = "system"


@dataclass
class ConversationMessage:
    """A single conversation message."""
    role: MessageRole
    content: str
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ConversationMessage:
        return cls(
            role=MessageRole(d.get("role", "human")),
            content=d.get("content", ""),
            timestamp=d.get("timestamp", 0.0),
        )


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Estimate token count from text length.

    A simple heuristic: ~4 characters per token for English text.
    Production code would use a proper tokenizer.
    """
    return max(1, len(text) // chars_per_token)


def estimate_messages_tokens(messages: list[ConversationMessage]) -> int:
    """Estimate total tokens across a list of messages."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg.content)
        # Each message has ~4 tokens of overhead (role markers)
        total += 4
    return total


class ConversationSummaryBuffer:
    """Buffer with summarizer for storing conversation memory.

    Provides a running summary of the conversation together with the most
    recent messages under the constraint that the total number of tokens
    in the conversation does not exceed a certain limit.
    """

    def __init__(
        self,
        max_token_limit: int = 2000,
        *,
        summarizer: Callable[[list[ConversationMessage], str], str] | None = None,
        memory_key: str = "history",
    ) -> None:
        self.max_token_limit = max_token_limit
        self.summarizer = summarizer or self._default_summarizer
        self.memory_key = memory_key
        self._messages: deque[ConversationMessage] = deque()
        self._moving_summary: str = ""
        self._lock = threading.Lock()

    @staticmethod
    def _default_summarizer(
        messages: list[ConversationMessage],
        existing_summary: str,
    ) -> str:
        """Default summarizer: concatenate message contents.

        In production, this would call an LLM. For now, we produce a
        simple text summary of the pruned messages.
        """
        parts = []
        if existing_summary:
            parts.append(existing_summary)
        for msg in messages:
            parts.append(f"{msg.role.value}: {msg.content}")
        return "\n".join(parts)

    @property
    def messages(self) -> list[ConversationMessage]:
        """Current messages in the buffer."""
        with self._lock:
            return list(self._messages)

    @property
    def moving_summary(self) -> str:
        """The running summary of pruned messages."""
        with self._lock:
            return self._moving_summary

    @property
    def buffer(self) -> str:
        """String buffer of memory (summary + messages)."""
        return self.load_memory_variables({})[self.memory_key]

    @property
    def memory_variables(self) -> list[str]:
        """Will always return list of memory variables."""
        return [self.memory_key]

    @property
    def message_count(self) -> int:
        with self._lock:
            return len(self._messages)

    @property
    def token_count(self) -> int:
        """Current estimated token count of buffer (summary + messages)."""
        with self._lock:
            total = estimate_tokens(self._moving_summary) if self._moving_summary else 0
            total += estimate_messages_tokens(list(self._messages))
            return total

    def save_context(
        self,
        human_input: str,
        ai_output: str,
    ) -> None:
        """Save a conversation turn to the buffer, then prune."""
        with self._lock:
            self._messages.append(
                ConversationMessage(role=MessageRole.HUMAN, content=human_input)
            )
            self._messages.append(
                ConversationMessage(role=MessageRole.AI, content=ai_output)
            )
            self._prune_locked()

    def add_message(self, message: ConversationMessage) -> None:
        """Add a single message to the buffer, then prune."""
        with self._lock:
            self._messages.append(message)
            self._prune_locked()

    def _prune_locked(self) -> None:
        """Prune buffer if it exceeds max token limit (caller holds lock)."""
        buffer = list(self._messages)
        curr_length = estimate_messages_tokens(buffer)
        if curr_length <= self.max_token_limit:
            return

        pruned: list[ConversationMessage] = []
        while buffer and curr_length > self.max_token_limit:
            pruned.append(buffer.pop(0))
            curr_length = estimate_messages_tokens(buffer)

        if pruned:
            self._moving_summary = self.summarizer(pruned, self._moving_summary)
            self._messages = deque(buffer)

    def prune(self) -> None:
        """Prune buffer if it exceeds max token limit."""
        with self._lock:
            self._prune_locked()

    def load_memory_variables(
        self,
        inputs: dict[str, Any] | None = None,
        *,
        return_messages: bool = False,
    ) -> dict[str, Any]:
        """Return history buffer (summary + messages)."""
        with self._lock:
            messages = list(self._messages)
            summary = self._moving_summary

        if return_messages:
            result: list[Any] = []
            if summary:
                result.append(
                    ConversationMessage(
                        role=MessageRole.SYSTEM,
                        content=summary,
                    )
                )
            result.extend(messages)
            return {self.memory_key: result}

        parts = []
        if summary:
            parts.append(f"Summary: {summary}")
        for msg in messages:
            prefix = msg.role.value.capitalize()
            parts.append(f"{prefix}: {msg.content}")
        return {self.memory_key: "\n".join(parts)}

    def clear(self) -> None:
        """Clear memory contents."""
        with self._lock:
            self._messages.clear()
            self._moving_summary = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to dict."""
        with self._lock:
            return {
                "max_token_limit": self.max_token_limit,
                "memory_key": self.memory_key,
                "moving_summary": self._moving_summary,
                "messages": [m.to_dict() for m in self._messages],
            }

    @classmethod
    def from_dict(
        cls,
        d: dict[str, Any],
        *,
        summarizer: Callable[[list[ConversationMessage], str], str] | None = None,
    ) -> ConversationSummaryBuffer:
        """Restore state from dict."""
        buf = cls(
            max_token_limit=d.get("max_token_limit", 2000),
            memory_key=d.get("memory_key", "history"),
            summarizer=summarizer,
        )
        buf._moving_summary = d.get("moving_summary", "")
        for msg_dict in d.get("messages", []):
            buf._messages.append(ConversationMessage.from_dict(msg_dict))
        return buf
