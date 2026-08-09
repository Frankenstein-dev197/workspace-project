"""Channel system: external messaging platform abstraction.

Integrates DeerFlow channels pattern:
- Channel: abstract base for IM/messaging platform connections
  - start/stop lifecycle
  - send: outbound message delivery
  - send_file: file attachment upload
  - supports_streaming: platform capability flag
  - _send_with_retry: exponential backoff for delivery
- ChannelRunPolicy: per-channel run behavior configuration
  - is_interactive: whether clarification is allowed
  - default_recursion_limit: max agent steps per run
  - requires_bound_identity: whether sender must be bound
  - fire_and_forget: whether to return immediately on run start
  - serialize_thread_runs: whether to serialize same-thread turns
- InboundMessage / OutboundMessage: message envelopes
- ChannelManager: registry and lifecycle for multiple channels

Each channel connects to an external messaging platform and:
1. Receives messages, wraps them as InboundMessage, publishes to bus.
2. Subscribes to outbound messages and sends replies back to the platform.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class InboundMessageType(Enum):
    """Type of inbound message."""
    TEXT = "text"
    COMMAND = "command"
    FILE = "file"
    EVENT = "event"


@dataclass
class InboundMessage:
    """A message received from an external platform."""
    channel: str
    chat_id: str
    sender_id: str
    content: str
    message_type: InboundMessageType = InboundMessageType.TEXT
    thread_ts: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def message_id(self) -> str:
        """Stable ID for dedup (channel:chat:sender:content hash)."""
        import hashlib
        h = hashlib.sha256(
            f"{self.channel}:{self.chat_id}:{self.sender_id}:{self.content}".encode()
        ).hexdigest()[:16]
        return h


@dataclass
class OutboundMessage:
    """A message to send to an external platform."""
    channel: str
    chat_id: str
    content: str
    thread_ts: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResolvedAttachment:
    """A resolved file attachment ready for upload."""
    filename: str
    content: bytes
    mime_type: str = "application/octet-stream"
    path: str | None = None


@dataclass(frozen=True)
class ChannelRunPolicy:
    """Per-channel run behavior configuration.

    Declaring all knobs on one dataclass keeps the channel's run
    behavior in a single discoverable place and turns "add a new
    channel" into a one-row registration instead of touching multiple
    separate methods on the manager.
    """
    is_interactive: bool = True
    default_recursion_limit: int | None = None
    requires_bound_identity: bool = True
    fire_and_forget: bool = False
    serialize_thread_runs: bool = False

    @property
    def disable_clarification(self) -> bool:
        """When True, agent proceeds with best judgment instead of asking."""
        return not self.is_interactive


class Channel:
    """Base class for all IM channel implementations.

    Each channel connects to an external messaging platform and:
    1. Receives messages, wraps them as InboundMessage, publishes to bus.
    2. Subscribes to outbound messages and sends replies back to the platform.

    Subclasses must implement ``start``, ``stop``, and ``send``.
    """

    def __init__(
        self,
        name: str,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.config = config or {}
        self._running = False
        self._lock = threading.Lock()
        self._inbound_handler: Callable[[InboundMessage], None] | None = None
        self._stats = {
            "inbound_count": 0,
            "outbound_count": 0,
            "errors": 0,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def supports_streaming(self) -> bool:
        return False

    @property
    def run_policy(self) -> ChannelRunPolicy:
        return ChannelRunPolicy()

    def set_inbound_handler(
        self,
        handler: Callable[[InboundMessage], None],
    ) -> None:
        """Set the handler for inbound messages."""
        with self._lock:
            self._inbound_handler = handler

    def publish_inbound(self, msg: InboundMessage) -> None:
        """Publish an inbound message to the registered handler."""
        with self._lock:
            self._stats["inbound_count"] += 1
            handler = self._inbound_handler
        if handler:
            try:
                handler(msg)
            except Exception as e:
                logger.error("Inbound handler error on %s: %s", self.name, e)
                with self._lock:
                    self._stats["errors"] += 1

    def start(self) -> None:
        """Start listening for messages from the external platform."""
        with self._lock:
            if self._running:
                return
            self._running = True
        logger.info("Channel '%s' started", self.name)

    def stop(self) -> None:
        """Gracefully stop the channel."""
        with self._lock:
            if not self._running:
                return
            self._running = False
        logger.info("Channel '%s' stopped", self.name)

    def send(self, msg: OutboundMessage) -> bool:
        """Send a message back to the external platform.

        Override in subclasses. Returns True if sent successfully.
        """
        with self._lock:
            self._stats["outbound_count"] += 1
        return False

    def send_file(
        self,
        msg: OutboundMessage,
        attachment: ResolvedAttachment,
    ) -> bool:
        """Upload a single file attachment to the platform.

        Returns True if the upload succeeded, False otherwise.
        Default implementation returns False (no file upload support).
        """
        return False

    def send_with_retry(
        self,
        operation: Callable[[], bool],
        max_retries: int = 3,
        base_delay: float = 0.5,
    ) -> bool:
        """Execute an operation with exponential backoff retry."""
        for attempt in range(max_retries):
            try:
                if operation():
                    return True
            except Exception as e:
                logger.warning(
                    "Send retry %d/%d on %s: %s",
                    attempt + 1,
                    max_retries,
                    self.name,
                    e,
                )
                with self._lock:
                    self._stats["errors"] += 1
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
        return False

    def get_stats(self) -> dict[str, Any]:
        """Get channel statistics."""
        with self._lock:
            return dict(self._stats)


class ChannelManager:
    """Registry and lifecycle manager for multiple channels."""

    def __init__(self) -> None:
        self._channels: dict[str, Channel] = {}
        self._policies: dict[str, ChannelRunPolicy] = {}
        self._lock = threading.Lock()

    def register(
        self,
        channel: Channel,
        policy: ChannelRunPolicy | None = None,
    ) -> None:
        """Register a channel with optional run policy."""
        with self._lock:
            self._channels[channel.name] = channel
            self._policies[channel.name] = policy or channel.run_policy

    def unregister(self, name: str) -> None:
        """Unregister a channel."""
        with self._lock:
            channel = self._channels.pop(name, None)
            self._policies.pop(name, None)
        if channel and channel.is_running:
            channel.stop()

    def get(self, name: str) -> Channel | None:
        """Get a channel by name."""
        with self._lock:
            return self._channels.get(name)

    def get_policy(self, name: str) -> ChannelRunPolicy | None:
        """Get the run policy for a channel."""
        with self._lock:
            return self._policies.get(name)

    def list_channels(self) -> list[str]:
        """List registered channel names."""
        with self._lock:
            return list(self._channels.keys())

    def start_all(self) -> None:
        """Start all registered channels."""
        with self._lock:
            channels = list(self._channels.values())
        for channel in channels:
            try:
                channel.start()
            except Exception as e:
                logger.error("Failed to start channel %s: %s", channel.name, e)

    def stop_all(self) -> None:
        """Stop all registered channels."""
        with self._lock:
            channels = list(self._channels.values())
        for channel in channels:
            try:
                channel.stop()
            except Exception as e:
                logger.error("Failed to stop channel %s: %s", channel.name, e)

    def send(self, msg: OutboundMessage) -> bool:
        """Send a message via the appropriate channel."""
        with self._lock:
            channel = self._channels.get(msg.channel)
        if not channel:
            logger.error("Channel '%s' not registered", msg.channel)
            return False
        return channel.send(msg)

    def all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all channels."""
        with self._lock:
            channels = dict(self._channels)
        return {name: ch.get_stats() for name, ch in channels.items()}

    def channel_count(self) -> int:
        """Number of registered channels."""
        with self._lock:
            return len(self._channels)
