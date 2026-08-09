"""Server-Sent Events (SSE) stream for agent execution.

Based on browser-use SSE event patterns:
- SSEEvent: typed event with event_type, data, timestamp, id
- SSEEventType: standard event types (log, result, error, progress, stream_complete)
- SSEStream: produces and consumes SSE-formatted events
  - emit: produce an event to the stream
  - subscribe: register a callback for events
  - consume: generator that yields events as they arrive
  - to_sse_format: serialize event to SSE wire format (data: ...\n\n)
  - from_sse_format: parse SSE wire format back to event
  - close: signal stream completion

Useful for streaming agent execution progress to clients over HTTP,
or for inter-process event streaming.
"""

from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SSEEventType(str, Enum):
    """Standard event types for SSE streams."""
    LOG = "log"
    PROGRESS = "progress"
    RESULT = "result"
    ERROR = "error"
    STREAM_COMPLETE = "stream_complete"
    CUSTOM = "custom"


@dataclass
class SSEEvent:
    """A single SSE event."""
    event_type: SSEEventType = SSEEventType.CUSTOM
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_name: str | None = None  # Override event name for wire format

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "id": self.id,
        }

    def to_sse_format(self) -> str:
        """Serialize to SSE wire format.

        Format:
            id: <id>
            event: <event_name>
            data: <json data>

            (blank line terminates event)
        """
        lines: list[str] = []
        lines.append(f"id: {self.id}")
        event_name = self.event_name or self.event_type.value
        lines.append(f"event: {event_name}")
        data_str = json.dumps(self.data, default=str)
        lines.append(f"data: {data_str}")
        return "\n".join(lines) + "\n\n"

    @classmethod
    def from_sse_format(cls, raw: str) -> "SSEEvent":
        """Parse SSE wire format into an event.

        Args:
            raw: raw SSE text (may include trailing newlines)

        Returns:
            SSEEvent instance
        """
        event_id = str(uuid.uuid4())
        event_name: str | None = None
        data_str = ""

        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("id:"):
                event_id = line[3:].strip()
            elif line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_str = line[5:].strip()

        try:
            data = json.loads(data_str) if data_str else {}
        except json.JSONDecodeError:
            data = {"raw": data_str}

        # Map event name back to enum
        try:
            event_type = SSEEventType(event_name) if event_name else SSEEventType.CUSTOM
        except ValueError:
            event_type = SSEEventType.CUSTOM

        return cls(
            event_type=event_type,
            data=data,
            id=event_id,
            event_name=event_name,
        )


class SSEStream:
    """Thread-safe SSE event stream.

    Producers emit events; consumers subscribe or iterate.
    Supports multiple subscribers and bounded buffering.
    """

    def __init__(self, *, maxsize: int = 1000) -> None:
        self._buffer: queue.Queue[SSEEvent | None] = queue.Queue(maxsize=maxsize)
        self._subscribers: list[Callable[[SSEEvent], None]] = []
        self._lock = threading.Lock()
        self._closed = False
        self._event_count = 0

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def event_count(self) -> int:
        return self._event_count

    def emit(
        self,
        event_type: SSEEventType,
        data: dict[str, Any] | None = None,
        *,
        event_name: str | None = None,
        _allow_closed: bool = False,
    ) -> SSEEvent:
        """Emit an event to the stream.

        Returns the emitted event.
        Raises RuntimeError if stream is closed (unless _allow_closed).
        """
        if self._closed and not _allow_closed:
            raise RuntimeError("Cannot emit to closed stream")

        event = SSEEvent(
            event_type=event_type,
            data=data or {},
            event_name=event_name,
        )

        with self._lock:
            self._event_count += 1
            subscribers = list(self._subscribers)

        # Notify subscribers (outside lock to prevent deadlock)
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                pass  # Don't let subscriber errors break the stream

        # Add to buffer (blocks if full)
        self._buffer.put(event)
        return event

    def emit_log(self, message: str, level: str = "info") -> SSEEvent:
        """Emit a log event."""
        return self.emit(
            SSEEventType.LOG,
            {"message": message, "level": level},
        )

    def emit_progress(
        self,
        current: int,
        total: int,
        message: str = "",
    ) -> SSEEvent:
        """Emit a progress event."""
        return self.emit(
            SSEEventType.PROGRESS,
            {"current": current, "total": total, "message": message},
        )

    def emit_result(self, result: Any) -> SSEEvent:
        """Emit a result event."""
        return self.emit(
            SSEEventType.RESULT,
            {"result": result},
        )

    def emit_error(self, error: str, traceback: str = "") -> SSEEvent:
        """Emit an error event."""
        return self.emit(
            SSEEventType.ERROR,
            {"error": error, "traceback": traceback},
        )

    def subscribe(self, callback: Callable[[SSEEvent], None]) -> None:
        """Register a callback to be called for each event."""
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[SSEEvent], None]) -> None:
        """Remove a registered callback."""
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def consume(
        self,
        *,
        timeout: float | None = None,
    ) -> Iterator[SSEEvent]:
        """Iterate over events as they arrive.

        Yields events until stream is closed and buffer is empty.
        If timeout is set, stops after timeout seconds of no events.

        Args:
            timeout: seconds to wait for each event (None = block forever)
        """
        while True:
            try:
                event = self._buffer.get(timeout=timeout)
            except queue.Empty:
                return  # Timeout reached

            if event is None:
                # Sentinel signals stream end
                return
            yield event

    def consume_all(self) -> list[SSEEvent]:
        """Consume all buffered events without blocking."""
        events: list[SSEEvent] = []
        while True:
            try:
                event = self._buffer.get_nowait()
            except queue.Empty:
                break
            if event is None:
                break
            events.append(event)
        return events

    def to_sse_lines(self) -> Iterator[str]:
        """Consume events and yield SSE wire-format strings.

        Useful for HTTP response streaming.
        Yields strings until stream closes.
        """
        for event in self.consume():
            yield event.to_sse_format()

    def close(self) -> None:
        """Signal that the stream is complete.

        Adds a None sentinel to the buffer to unblock consumers.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True

        # Emit stream_complete event
        self.emit(
            SSEEventType.STREAM_COMPLETE,
            {"event_count": self._event_count},
            _allow_closed=True,
        )
        # Sentinel to unblock consumers
        self._buffer.put(None)

    def __enter__(self) -> "SSEStream":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def parse_sse_stream(raw_text: str) -> list[SSEEvent]:
    """Parse a raw SSE text stream into events.

    Events are separated by blank lines (\\n\\n).
    """
    events: list[SSEEvent] = []
    # Split on double newline (event separator)
    chunks = raw_text.split("\n\n")
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        events.append(SSEEvent.from_sse_format(chunk))
    return events
