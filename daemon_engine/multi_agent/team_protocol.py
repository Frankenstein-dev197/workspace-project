"""Team protocols: structured inter-agent request/response coordination.

Integrates learn-claude-code s16 team protocols pattern:
- ProtocolState: tracks in-flight protocol requests (shutdown, plan approval)
- ProtocolManager: manages request lifecycle and response matching
- request_shutdown: lead sends shutdown protocol request to teammate
- request_plan: lead asks teammate to submit a plan
- match_response: correlates response to request via request_id
- Type validation: response type must match request type
- Status tracking: pending → approved/rejected

Protocol message types:
- shutdown_request → shutdown_response
- plan_approval → plan_approval_response
- status_request → status_response

This enables structured coordination between lead and teammate agents
with request tracking, type validation, and duplicate prevention.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from daemon_engine.multi_agent.message_bus import MessageBus

logger = logging.getLogger(__name__)


class ProtocolType(str, Enum):
    """Types of protocol requests."""
    SHUTDOWN = "shutdown"
    PLAN_APPROVAL = "plan_approval"
    STATUS = "status"


class ProtocolStatus(str, Enum):
    """Status of a protocol request."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


# Request type → expected response type suffix
RESPONSE_SUFFIX = "_response"


@dataclass
class ProtocolState:
    """Tracks an in-flight protocol request."""
    request_id: str
    protocol_type: str
    sender: str
    target: str
    status: ProtocolStatus = ProtocolStatus.PENDING
    payload: str = ""
    created_at: float = field(default_factory=time.time)
    responded_at: float = 0.0
    response_payload: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "type": self.protocol_type,
            "sender": self.sender,
            "target": self.target,
            "status": self.status.value,
            "payload": self.payload,
            "created_at": self.created_at,
            "responded_at": self.responded_at,
            "response_payload": self.response_payload,
        }


def new_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{random.randint(0, 999999):06d}"


class ProtocolManager:
    """Manages protocol request/response lifecycle.

    Tracks in-flight requests, validates responses, and prevents
    duplicate processing. Works with MessageBus for delivery.
    """

    def __init__(self, bus: MessageBus | None = None) -> None:
        self._bus = bus
        self._pending: dict[str, ProtocolState] = {}
        self._completed: dict[str, ProtocolState] = {}
        self._stats = {
            "total_requests": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "total_expired": 0,
            "by_type": {},
        }

    @property
    def bus(self) -> MessageBus | None:
        return self._bus

    @bus.setter
    def bus(self, value: MessageBus) -> None:
        self._bus = value

    def _make_request(
        self,
        protocol_type: ProtocolType,
        sender: str,
        target: str,
        payload: str = "",
    ) -> ProtocolState:
        """Create and track a new protocol request."""
        request_id = new_request_id()
        state = ProtocolState(
            request_id=request_id,
            protocol_type=protocol_type.value,
            sender=sender,
            target=target,
            payload=payload,
        )
        self._pending[request_id] = state
        self._stats["total_requests"] += 1
        type_key = protocol_type.value
        self._stats["by_type"][type_key] = self._stats["by_type"].get(type_key, 0) + 1
        return state

    def request_shutdown(
        self,
        sender: str,
        target: str,
        reason: str = "",
    ) -> ProtocolState:
        """Send a shutdown request to a teammate."""
        state = self._make_request(ProtocolType.SHUTDOWN, sender, target, reason)
        if self._bus:
            self._bus.send(
                from_agent=sender,
                to_agent=target,
                content=reason or "Shutdown requested",
                msg_type="shutdown_request",
                metadata={"request_id": state.request_id},
            )
        logger.info("Shutdown request %s: %s → %s", state.request_id, sender, target)
        return state

    def request_plan(
        self,
        sender: str,
        target: str,
        task_description: str = "",
    ) -> ProtocolState:
        """Request a plan from a teammate."""
        state = self._make_request(ProtocolType.PLAN_APPROVAL, sender, target, task_description)
        if self._bus:
            self._bus.send(
                from_agent=sender,
                to_agent=target,
                content=task_description,
                msg_type="plan_request",
                metadata={"request_id": state.request_id},
            )
        logger.info("Plan request %s: %s → %s", state.request_id, sender, target)
        return state

    def request_status(
        self,
        sender: str,
        target: str,
        query: str = "",
    ) -> ProtocolState:
        """Request status from a teammate."""
        state = self._make_request(ProtocolType.STATUS, sender, target, query)
        if self._bus:
            self._bus.send(
                from_agent=sender,
                to_agent=target,
                content=query,
                msg_type="status_request",
                metadata={"request_id": state.request_id},
            )
        logger.info("Status request %s: %s → %s", state.request_id, sender, target)
        return state

    def match_response(
        self,
        response_type: str,
        request_id: str,
        approve: bool,
        response_payload: str = "",
    ) -> bool:
        """Correlate a response to the original request.

        Validates that response_type matches the request type.
        Returns True if matched successfully, False otherwise.
        """
        state = self._pending.get(request_id)
        if not state:
            logger.warning("Unknown request_id: %s", request_id)
            return False

        expected_response = f"{state.protocol_type}{RESPONSE_SUFFIX}"
        if response_type != expected_response:
            logger.warning(
                "Type mismatch: expected %s, got %s",
                expected_response,
                response_type,
            )
            return False

        if state.status != ProtocolStatus.PENDING:
            logger.warning("Request %s already %s, ignoring", request_id, state.status)
            return False

        state.status = ProtocolStatus.APPROVED if approve else ProtocolStatus.REJECTED
        state.responded_at = time.time()
        state.response_payload = response_payload

        self._pending.pop(request_id)
        self._completed[request_id] = state

        if approve:
            self._stats["total_approved"] += 1
        else:
            self._stats["total_rejected"] += 1

        logger.info(
            "Protocol %s %s: %s (%s)",
            state.protocol_type,
            state.status.value,
            request_id,
            state.sender,
        )
        return True

    def consume_responses(self, agent: str) -> list[dict[str, Any]]:
        """Read agent's inbox and route protocol responses.

        Returns all messages (including non-protocol) for injection.
        """
        if not self._bus:
            return []
        msgs = self._bus.read_inbox(agent)
        if not msgs:
            return []
        for msg in msgs:
            meta = msg.metadata
            req_id = meta.get("request_id", "")
            msg_type = msg.msg_type
            if req_id and msg_type.endswith(RESPONSE_SUFFIX):
                approve = meta.get("approve", False)
                self.match_response(msg_type, req_id, approve, msg.content)
        return [m.to_dict() for m in msgs]

    def get_request(self, request_id: str) -> ProtocolState | None:
        """Get a request by ID (pending or completed)."""
        return self._pending.get(request_id) or self._completed.get(request_id)

    def list_pending(self) -> list[ProtocolState]:
        """List all pending requests."""
        return list(self._pending.values())

    def list_completed(self) -> list[ProtocolState]:
        """List all completed requests."""
        return list(self._completed.values())

    def expire_stale(self, timeout: float = 300.0) -> int:
        """Expire requests that have been pending too long."""
        now = time.time()
        expired = 0
        for req_id, state in list(self._pending.items()):
            if now - state.created_at > timeout:
                state.status = ProtocolStatus.EXPIRED
                state.responded_at = now
                self._pending.pop(req_id)
                self._completed[req_id] = state
                self._stats["total_expired"] += 1
                expired += 1
        return expired

    def stats(self) -> dict[str, Any]:
        return {
            **self._stats,
            "pending": len(self._pending),
            "completed": len(self._completed),
        }

    def clear(self) -> None:
        """Clear all tracked requests."""
        self._pending.clear()
        self._completed.clear()
