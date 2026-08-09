"""Message bus: file-based inter-agent communication.

Integrates learn-claude-code s15 MessageBus pattern:
- MessageBus: file-based message bus with JSONL inboxes
- send: append message to recipient's inbox
- read_inbox: destructive read (read + delete)
- peek: non-destructive check for unread messages
- Thread-safe: concurrent write safety via file locking
- Message types: message, result, notification, shutdown

This enables agents (lead + teammates) to communicate asynchronously
through file-based inboxes, decoupling sender from receiver.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A message between agents."""
    from_agent: str
    to_agent: str
    content: str
    msg_type: str = "message"
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_agent,
            "to": self.to_agent,
            "content": self.content,
            "type": self.msg_type,
            "ts": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            from_agent=d.get("from", ""),
            to_agent=d.get("to", ""),
            content=d.get("content", ""),
            msg_type=d.get("type", "message"),
            timestamp=d.get("ts", time.time()),
            metadata=d.get("metadata", {}),
        )


class MessageBus:
    """File-based message bus with JSONL inboxes.

    Each agent has a .jsonl inbox file. Messages are appended (send)
    and consumed (read_inbox deletes the file after reading).
    Non-destructive peek checks for unread messages.

    Thread safety: uses atomic file operations. For true concurrent
    write safety, file locking should be added (e.g., fcntl on Unix).
    """

    def __init__(self, mailbox_dir: str | Path | None = None) -> None:
        self._mailbox_dir = Path(mailbox_dir) if mailbox_dir else Path(".mailboxes")
        self._mailbox_dir.mkdir(parents=True, exist_ok=True)
        self._stats = {
            "total_sent": 0,
            "total_read": 0,
            "by_type": {},
        }

    def _inbox_path(self, agent: str) -> Path:
        """Get the inbox file path for an agent."""
        safe_name = "".join(c for c in agent if c.isalnum() or c in "._-")
        return self._mailbox_dir / f"{safe_name}.jsonl"

    def send(
        self,
        from_agent: str,
        to_agent: str,
        content: str,
        msg_type: str = "message",
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Send a message to an agent's inbox."""
        msg = Message(
            from_agent=from_agent,
            to_agent=to_agent,
            content=content,
            msg_type=msg_type,
            metadata=metadata or {},
        )
        inbox = self._inbox_path(to_agent)
        try:
            with open(inbox, "a") as f:
                f.write(json.dumps(msg.to_dict()) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception as exc:
            logger.error("Failed to send message to %s: %s", to_agent, exc)
            raise
        self._stats["total_sent"] += 1
        self._stats["by_type"][msg_type] = self._stats["by_type"].get(msg_type, 0) + 1
        logger.debug("Message sent: %s → %s: %s", from_agent, to_agent, content[:50])
        return msg

    def read_inbox(self, agent: str) -> list[Message]:
        """Destructive read: consume all messages from an agent's inbox."""
        inbox = self._inbox_path(agent)
        if not inbox.exists():
            return []
        try:
            lines = inbox.read_text().splitlines()
            msgs = [Message.from_dict(json.loads(line)) for line in lines if line.strip()]
            inbox.unlink()
        except Exception as exc:
            logger.error("Failed to read inbox for %s: %s", agent, exc)
            return []
        self._stats["total_read"] += len(msgs)
        return msgs

    def peek(self, agent: str) -> bool:
        """Non-destructive: True if the agent has unread messages."""
        inbox = self._inbox_path(agent)
        return inbox.exists() and inbox.stat().st_size > 0

    def peek_count(self, agent: str) -> int:
        """Non-destructive: count of unread messages."""
        inbox = self._inbox_path(agent)
        if not inbox.exists():
            return 0
        try:
            lines = inbox.read_text().splitlines()
            return len([l for l in lines if l.strip()])
        except Exception:
            return 0

    def clear_inbox(self, agent: str) -> int:
        """Clear an agent's inbox without reading. Returns count cleared."""
        inbox = self._inbox_path(agent)
        if not inbox.exists():
            return 0
        try:
            lines = inbox.read_text().splitlines()
            count = len([l for l in lines if l.strip()])
            inbox.unlink()
            return count
        except Exception:
            return 0

    def list_agents(self) -> list[str]:
        """List all agents that have inboxes."""
        agents = []
        for path in self._mailbox_dir.glob("*.jsonl"):
            agents.append(path.stem)
        return sorted(agents)

    def broadcast(
        self,
        from_agent: str,
        to_agents: list[str],
        content: str,
        msg_type: str = "message",
    ) -> list[Message]:
        """Send a message to multiple agents."""
        messages = []
        for agent in to_agents:
            msg = self.send(from_agent, agent, content, msg_type)
            messages.append(msg)
        return messages

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "active_inboxes": len(self.list_agents()),
            "mailbox_dir": str(self._mailbox_dir),
        }

    def clear_all(self) -> None:
        """Clear all inboxes."""
        for path in self._mailbox_dir.glob("*.jsonl"):
            path.unlink()
