"""Tests for team protocol manager."""

import time
from pathlib import Path

import pytest

from daemon_engine.multi_agent.team_protocol import (
    ProtocolManager,
    ProtocolState,
    ProtocolType,
    ProtocolStatus,
    new_request_id,
)
from daemon_engine.multi_agent.message_bus import MessageBus


class TestProtocolState:
    def test_creation(self):
        state = ProtocolState(
            request_id="req_001",
            protocol_type="shutdown",
            sender="lead",
            target="worker",
        )
        assert state.request_id == "req_001"
        assert state.status == ProtocolStatus.PENDING

    def test_to_dict(self):
        state = ProtocolState(
            request_id="req_001",
            protocol_type="shutdown",
            sender="lead",
            target="worker",
        )
        d = state.to_dict()
        assert d["request_id"] == "req_001"
        assert d["status"] == "pending"


class TestNewRequestId:
    def test_format(self):
        req_id = new_request_id()
        assert req_id.startswith("req_")

    def test_unique(self):
        ids = {new_request_id() for _ in range(100)}
        assert len(ids) == 100


class TestProtocolManager:
    def test_creation(self):
        manager = ProtocolManager()
        assert len(manager.list_pending()) == 0

    def test_request_shutdown(self):
        manager = ProtocolManager()
        state = manager.request_shutdown("lead", "worker", "Shutting down")
        assert state.protocol_type == "shutdown"
        assert state.sender == "lead"
        assert state.target == "worker"
        assert state.status == ProtocolStatus.PENDING
        assert len(manager.list_pending()) == 1

    def test_request_plan(self):
        manager = ProtocolManager()
        state = manager.request_plan("lead", "worker", "Build feature X")
        assert state.protocol_type == "plan_approval"
        assert state.payload == "Build feature X"

    def test_request_status(self):
        manager = ProtocolManager()
        state = manager.request_status("lead", "worker", "progress?")
        assert state.protocol_type == "status"

    def test_match_response_approve(self):
        manager = ProtocolManager()
        state = manager.request_shutdown("lead", "worker")
        result = manager.match_response("shutdown_response", state.request_id, approve=True)
        assert result is True
        assert state.status == ProtocolStatus.APPROVED
        assert len(manager.list_pending()) == 0
        assert len(manager.list_completed()) == 1

    def test_match_response_reject(self):
        manager = ProtocolManager()
        state = manager.request_shutdown("lead", "worker")
        result = manager.match_response("shutdown_response", state.request_id, approve=False)
        assert result is True
        assert state.status == ProtocolStatus.REJECTED

    def test_match_response_unknown_id(self):
        manager = ProtocolManager()
        result = manager.match_response("shutdown_response", "unknown", approve=True)
        assert result is False

    def test_match_response_type_mismatch(self):
        manager = ProtocolManager()
        state = manager.request_shutdown("lead", "worker")
        result = manager.match_response("plan_approval_response", state.request_id, approve=True)
        assert result is False

    def test_match_response_duplicate(self):
        manager = ProtocolManager()
        state = manager.request_shutdown("lead", "worker")
        manager.match_response("shutdown_response", state.request_id, approve=True)
        result = manager.match_response("shutdown_response", state.request_id, approve=True)
        assert result is False

    def test_get_request(self):
        manager = ProtocolManager()
        state = manager.request_shutdown("lead", "worker")
        retrieved = manager.get_request(state.request_id)
        assert retrieved is not None
        assert retrieved.request_id == state.request_id

    def test_get_request_nonexistent(self):
        manager = ProtocolManager()
        assert manager.get_request("nonexistent") is None

    def test_with_message_bus(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        manager = ProtocolManager(bus=bus)
        manager.request_shutdown("lead", "worker", "Bye")
        assert bus.peek("worker") is True
        msgs = bus.read_inbox("worker")
        assert len(msgs) == 1
        assert msgs[0].msg_type == "shutdown_request"

    def test_consume_responses(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        manager = ProtocolManager(bus=bus)
        state = manager.request_shutdown("lead", "worker")
        bus.send("worker", "lead", "OK shutting down", "shutdown_response",
                 metadata={"request_id": state.request_id, "approve": True})
        msgs = manager.consume_responses("lead")
        assert len(msgs) == 1
        assert state.status == ProtocolStatus.APPROVED

    def test_consume_responses_empty(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        manager = ProtocolManager(bus=bus)
        msgs = manager.consume_responses("lead")
        assert msgs == []

    def test_consume_responses_no_bus(self):
        manager = ProtocolManager()
        msgs = manager.consume_responses("lead")
        assert msgs == []

    def test_expire_stale(self):
        manager = ProtocolManager()
        state = manager.request_shutdown("lead", "worker")
        state.created_at = time.time() - 400
        expired = manager.expire_stale(timeout=300.0)
        assert expired == 1
        assert state.status == ProtocolStatus.EXPIRED
        assert len(manager.list_pending()) == 0

    def test_expire_not_stale(self):
        manager = ProtocolManager()
        manager.request_shutdown("lead", "worker")
        expired = manager.expire_stale(timeout=300.0)
        assert expired == 0

    def test_stats(self):
        manager = ProtocolManager()
        manager.request_shutdown("lead", "worker1")
        manager.request_plan("lead", "worker2")
        state = manager.request_shutdown("lead", "worker3")
        manager.match_response("shutdown_response", state.request_id, approve=True)
        stats = manager.stats()
        assert stats["total_requests"] == 3
        assert stats["total_approved"] == 1
        assert stats["by_type"]["shutdown"] == 2
        assert stats["by_type"]["plan_approval"] == 1

    def test_clear(self):
        manager = ProtocolManager()
        manager.request_shutdown("lead", "worker")
        manager.clear()
        assert len(manager.list_pending()) == 0
        assert len(manager.list_completed()) == 0

    def test_bus_setter(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        manager = ProtocolManager()
        manager.bus = bus
        assert manager.bus is bus

    def test_response_payload(self):
        manager = ProtocolManager()
        state = manager.request_plan("lead", "worker", "Build X")
        manager.match_response(
            "plan_approval_response",
            state.request_id,
            approve=True,
            response_payload="Here is my plan: 1. Design 2. Build 3. Test",
        )
        assert "Here is my plan" in state.response_payload

    def test_full_shutdown_lifecycle(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        manager = ProtocolManager(bus=bus)
        state = manager.request_shutdown("lead", "worker", "Done with work")
        assert bus.peek("worker") is True
        inbox = bus.read_inbox("worker")
        assert len(inbox) == 1
        req_id = inbox[0].metadata["request_id"]
        assert req_id == state.request_id
        bus.send("worker", "lead", "Acknowledged", "shutdown_response",
                 metadata={"request_id": req_id, "approve": True})
        manager.consume_responses("lead")
        assert state.status == ProtocolStatus.APPROVED
