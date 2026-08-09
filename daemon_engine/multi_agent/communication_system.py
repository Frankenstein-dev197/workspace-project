"""Communication System: inter-agent messaging and coordination.

Inspired by Ruflo's federated communication and learn-claude-code's team
protocols. Provides pub/sub messaging, direct messaging, and shared state
between agents.
"""

from __future__ import annotations

import logging
import queue
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class MessageType(Enum):
    DIRECT = "direct"
    BROADCAST = "broadcast"
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    STATUS_UPDATE = "status_update"


@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: str | None = None
    msg_type: MessageType = MessageType.DIRECT
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    reply_to: str | None = None
    timestamp: float = field(default_factory=time.time)

    @property
    def is_broadcast(self) -> bool:
        return self.recipient_id is None or self.msg_type == MessageType.BROADCAST


class CommunicationSystem:
    """Manages message routing between agents via channels and queues."""

    def __init__(self) -> None:
        self._channels: dict[str, queue.Queue] = {}
        self._subscribers: dict[str, list[Callable[[Message], None]]] = {}
        self._message_log: list[Message] = []
        self._shared_state: dict[str, Any] = {}

    def register_agent(self, agent_id: str) -> None:
        if agent_id not in self._channels:
            self._channels[agent_id] = queue.Queue()
            self._subscribers[agent_id] = []
            logger.info("Registered agent %s in communication system", agent_id)

    def unregister_agent(self, agent_id: str) -> None:
        self._channels.pop(agent_id, None)
        self._subscribers.pop(agent_id, None)

    def send(self, message: Message) -> bool:
        self._message_log.append(message)
        if message.is_broadcast:
            for agent_id in self._channels:
                if agent_id != message.sender_id:
                    self._channels[agent_id].put(message)
                    self._notify_subscribers(agent_id, message)
            logger.debug("Broadcast from %s to all agents", message.sender_id)
            return True
        if message.recipient_id and message.recipient_id in self._channels:
            self._channels[message.recipient_id].put(message)
            self._notify_subscribers(message.recipient_id, message)
            logger.debug("Message from %s to %s", message.sender_id, message.recipient_id)
            return True
        logger.warning("Recipient %s not found", message.recipient_id)
        return False

    def receive(self, agent_id: str, timeout: float = 0.1) -> Message | None:
        ch = self._channels.get(agent_id)
        if not ch:
            return None
        try:
            return ch.get(timeout=timeout)
        except queue.Empty:
            return None

    def subscribe(self, agent_id: str, callback: Callable[[Message], None]) -> None:
        if agent_id not in self._subscribers:
            self._subscribers[agent_id] = []
        self._subscribers[agent_id].append(callback)

    def _notify_subscribers(self, agent_id: str, message: Message) -> None:
        for callback in self._subscribers.get(agent_id, []):
            try:
                callback(message)
            except Exception as exc:
                logger.error("Subscriber callback error: %s", exc)

    def set_shared_state(self, key: str, value: Any) -> None:
        self._shared_state[key] = value

    def get_shared_state(self, key: str, default: Any = None) -> Any:
        return self._shared_state.get(key, default)

    def get_message_log(self, agent_id: str | None = None) -> list[Message]:
        if agent_id:
            return [m for m in self._message_log if m.sender_id == agent_id or m.recipient_id == agent_id]
        return self._message_log

    def clear(self) -> None:
        for ch in self._channels.values():
            while not ch.empty():
                ch.get()
        self._message_log.clear()
        self._shared_state.clear()
