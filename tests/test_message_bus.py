"""Tests for message bus."""

import json
from pathlib import Path

import pytest

from daemon_engine.multi_agent.message_bus import MessageBus, Message


class TestMessage:
    def test_creation(self):
        msg = Message(from_agent="lead", to_agent="worker1", content="Do task X")
        assert msg.from_agent == "lead"
        assert msg.to_agent == "worker1"
        assert msg.content == "Do task X"
        assert msg.msg_type == "message"

    def test_to_dict(self):
        msg = Message(from_agent="a", to_agent="b", content="hello", msg_type="result")
        d = msg.to_dict()
        assert d["from"] == "a"
        assert d["to"] == "b"
        assert d["content"] == "hello"
        assert d["type"] == "result"

    def test_from_dict(self):
        d = {"from": "a", "to": "b", "content": "hello", "type": "result", "ts": 123.0}
        msg = Message.from_dict(d)
        assert msg.from_agent == "a"
        assert msg.to_agent == "b"
        assert msg.content == "hello"
        assert msg.msg_type == "result"
        assert msg.timestamp == 123.0

    def test_roundtrip(self):
        msg = Message(from_agent="a", to_agent="b", content="test", metadata={"key": "value"})
        d = msg.to_dict()
        msg2 = Message.from_dict(d)
        assert msg2.from_agent == msg.from_agent
        assert msg2.to_agent == msg.to_agent
        assert msg2.content == msg.content
        assert msg2.metadata == msg.metadata


class TestMessageBus:
    def test_creation(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        assert len(bus.list_agents()) == 0

    def test_send_and_read(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        bus.send("lead", "worker", "Do task")
        msgs = bus.read_inbox("worker")
        assert len(msgs) == 1
        assert msgs[0].content == "Do task"
        assert msgs[0].from_agent == "lead"

    def test_read_empty_inbox(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        msgs = bus.read_inbox("nonexistent")
        assert msgs == []

    def test_read_is_destructive(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        bus.send("lead", "worker", "message")
        bus.read_inbox("worker")
        msgs = bus.read_inbox("worker")
        assert msgs == []

    def test_peek(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        assert bus.peek("worker") is False
        bus.send("lead", "worker", "message")
        assert bus.peek("worker") is True

    def test_peek_after_read(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        bus.send("lead", "worker", "message")
        bus.read_inbox("worker")
        assert bus.peek("worker") is False

    def test_peek_count(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        bus.send("lead", "worker", "msg1")
        bus.send("lead", "worker", "msg2")
        assert bus.peek_count("worker") == 2

    def test_multiple_messages(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        bus.send("lead", "worker", "msg1")
        bus.send("lead", "worker", "msg2")
        bus.send("lead", "worker", "msg3")
        msgs = bus.read_inbox("worker")
        assert len(msgs) == 3
        contents = [m.content for m in msgs]
        assert "msg1" in contents
        assert "msg3" in contents

    def test_message_types(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        bus.send("lead", "worker", "result data", msg_type="result")
        bus.send("lead", "worker", "notification", msg_type="notification")
        msgs = bus.read_inbox("worker")
        assert msgs[0].msg_type == "result"
        assert msgs[1].msg_type == "notification"

    def test_clear_inbox(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        bus.send("lead", "worker", "msg1")
        bus.send("lead", "worker", "msg2")
        count = bus.clear_inbox("worker")
        assert count == 2
        assert bus.peek("worker") is False

    def test_clear_empty_inbox(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        count = bus.clear_inbox("nonexistent")
        assert count == 0

    def test_list_agents(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        bus.send("lead", "worker1", "msg")
        bus.send("lead", "worker2", "msg")
        agents = bus.list_agents()
        assert "worker1" in agents
        assert "worker2" in agents

    def test_broadcast(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        msgs = bus.broadcast("lead", ["worker1", "worker2", "worker3"], "team update")
        assert len(msgs) == 3
        assert bus.peek("worker1") is True
        assert bus.peek("worker2") is True
        assert bus.peek("worker3") is True

    def test_stats(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        bus.send("lead", "worker", "msg1")
        bus.send("lead", "worker", "msg2", msg_type="result")
        stats = bus.stats()
        assert stats["total_sent"] == 2
        assert stats["by_type"]["message"] == 1
        assert stats["by_type"]["result"] == 1

    def test_stats_after_read(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        bus.send("lead", "worker", "msg1")
        bus.send("lead", "worker", "msg2")
        bus.read_inbox("worker")
        stats = bus.stats()
        assert stats["total_read"] == 2

    def test_clear_all(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        bus.send("lead", "worker1", "msg")
        bus.send("lead", "worker2", "msg")
        bus.clear_all()
        assert len(bus.list_agents()) == 0

    def test_safe_agent_name(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        bus.send("lead", "worker/../../etc", "msg")
        inbox_files = list((tmp_path / "mailboxes").glob("*.jsonl"))
        assert all("../" not in f.name for f in inbox_files)

    def test_metadata(self, tmp_path):
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")
        bus.send("lead", "worker", "msg", metadata={"task_id": "task_123"})
        msgs = bus.read_inbox("worker")
        assert msgs[0].metadata["task_id"] == "task_123"

    def test_persistence(self, tmp_path):
        mailbox = tmp_path / "mailboxes"
        bus1 = MessageBus(mailbox_dir=mailbox)
        bus1.send("lead", "worker", "persistent message")
        bus2 = MessageBus(mailbox_dir=mailbox)
        msgs = bus2.read_inbox("worker")
        assert len(msgs) == 1
        assert msgs[0].content == "persistent message"

    def test_concurrent_sends(self, tmp_path):
        import threading
        bus = MessageBus(mailbox_dir=tmp_path / "mailboxes")

        def sender(idx):
            bus.send(f"agent{idx}", "worker", f"msg_{idx}")

        threads = [threading.Thread(target=sender, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        msgs = bus.read_inbox("worker")
        assert len(msgs) == 10
