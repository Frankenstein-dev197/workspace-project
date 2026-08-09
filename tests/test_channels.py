"""Tests for channel system."""

import pytest

from daemon_engine.multi_agent.channels import (
    Channel,
    ChannelManager,
    ChannelRunPolicy,
    InboundMessage,
    InboundMessageType,
    OutboundMessage,
    ResolvedAttachment,
)


class TestInboundMessageType:
    def test_values(self):
        assert InboundMessageType.TEXT.value == "text"
        assert InboundMessageType.COMMAND.value == "command"
        assert InboundMessageType.FILE.value == "file"
        assert InboundMessageType.EVENT.value == "event"


class TestInboundMessage:
    def test_creation(self):
        msg = InboundMessage(
            channel="slack",
            chat_id="C123",
            sender_id="U456",
            content="hello",
        )
        assert msg.channel == "slack"
        assert msg.content == "hello"
        assert msg.message_type == InboundMessageType.TEXT

    def test_message_id_stable(self):
        msg1 = InboundMessage("slack", "C1", "U1", "hello")
        msg2 = InboundMessage("slack", "C1", "U1", "hello")
        assert msg1.message_id == msg2.message_id

    def test_message_id_differs(self):
        msg1 = InboundMessage("slack", "C1", "U1", "hello")
        msg2 = InboundMessage("slack", "C1", "U1", "world")
        assert msg1.message_id != msg2.message_id

    def test_default_timestamp(self):
        msg = InboundMessage("slack", "C1", "U1", "hello")
        assert msg.timestamp > 0

    def test_attachments_default_empty(self):
        msg = InboundMessage("slack", "C1", "U1", "hello")
        assert msg.attachments == []


class TestOutboundMessage:
    def test_creation(self):
        msg = OutboundMessage(
            channel="slack",
            chat_id="C123",
            content="reply",
        )
        assert msg.channel == "slack"
        assert msg.content == "reply"

    def test_thread_ts(self):
        msg = OutboundMessage(
            channel="slack",
            chat_id="C1",
            content="reply",
            thread_ts="1234567890.123",
        )
        assert msg.thread_ts == "1234567890.123"


class TestResolvedAttachment:
    def test_creation(self):
        att = ResolvedAttachment(
            filename="report.pdf",
            content=b"%PDF-1.4",
            mime_type="application/pdf",
        )
        assert att.filename == "report.pdf"
        assert att.content == b"%PDF-1.4"
        assert att.mime_type == "application/pdf"

    def test_defaults(self):
        att = ResolvedAttachment(filename="file.txt", content=b"data")
        assert att.mime_type == "application/octet-stream"
        assert att.path is None


class TestChannelRunPolicy:
    def test_defaults(self):
        policy = ChannelRunPolicy()
        assert policy.is_interactive is True
        assert policy.default_recursion_limit is None
        assert policy.requires_bound_identity is True
        assert policy.fire_and_forget is False
        assert policy.serialize_thread_runs is False

    def test_disable_clarification(self):
        interactive = ChannelRunPolicy(is_interactive=True)
        assert interactive.disable_clarification is False

        webhook = ChannelRunPolicy(is_interactive=False)
        assert webhook.disable_clarification is True

    def test_webhook_policy(self):
        policy = ChannelRunPolicy(
            is_interactive=False,
            default_recursion_limit=250,
            requires_bound_identity=False,
            fire_and_forget=True,
        )
        assert policy.is_interactive is False
        assert policy.default_recursion_limit == 250
        assert policy.requires_bound_identity is False
        assert policy.fire_and_forget is True

    def test_frozen(self):
        policy = ChannelRunPolicy()
        with pytest.raises(Exception):
            policy.is_interactive = False  # type: ignore[misc]


class TestChannel:
    def test_creation(self):
        ch = Channel("slack")
        assert ch.name == "slack"
        assert ch.is_running is False
        assert ch.supports_streaming is False

    def test_start(self):
        ch = Channel("slack")
        ch.start()
        assert ch.is_running is True

    def test_start_idempotent(self):
        ch = Channel("slack")
        ch.start()
        ch.start()  # second call should be no-op
        assert ch.is_running is True

    def test_stop(self):
        ch = Channel("slack")
        ch.start()
        ch.stop()
        assert ch.is_running is False

    def test_stop_idempotent(self):
        ch = Channel("slack")
        ch.stop()  # not started
        assert ch.is_running is False

    def test_default_run_policy(self):
        ch = Channel("slack")
        policy = ch.run_policy
        assert isinstance(policy, ChannelRunPolicy)
        assert policy.is_interactive is True

    def test_set_inbound_handler(self):
        ch = Channel("slack")
        received = []
        ch.set_inbound_handler(lambda msg: received.append(msg))
        ch.start()
        msg = InboundMessage("slack", "C1", "U1", "hello")
        ch.publish_inbound(msg)
        assert len(received) == 1
        assert received[0].content == "hello"

    def test_publish_inbound_without_handler(self):
        ch = Channel("slack")
        ch.start()
        msg = InboundMessage("slack", "C1", "U1", "hello")
        ch.publish_inbound(msg)  # should not error

    def test_publish_inbound_increments_stats(self):
        ch = Channel("slack")
        ch.set_inbound_handler(lambda msg: None)
        ch.start()
        ch.publish_inbound(InboundMessage("slack", "C1", "U1", "hello"))
        ch.publish_inbound(InboundMessage("slack", "C1", "U1", "world"))
        stats = ch.get_stats()
        assert stats["inbound_count"] == 2

    def test_handler_error_counted(self):
        ch = Channel("slack")
        def bad_handler(msg):
            raise ValueError("oops")
        ch.set_inbound_handler(bad_handler)
        ch.start()
        ch.publish_inbound(InboundMessage("slack", "C1", "U1", "hello"))
        stats = ch.get_stats()
        assert stats["errors"] == 1

    def test_send_increments_stats(self):
        ch = Channel("slack")
        ch.send(OutboundMessage("slack", "C1", "reply"))
        stats = ch.get_stats()
        assert stats["outbound_count"] == 1

    def test_send_file_default_false(self):
        ch = Channel("slack")
        msg = OutboundMessage("slack", "C1", "reply")
        att = ResolvedAttachment("file.txt", b"data")
        assert ch.send_file(msg, att) is False

    def test_send_with_retry_success(self):
        ch = Channel("slack")
        attempts = []
        def op():
            attempts.append(1)
            return True
        assert ch.send_with_retry(op, max_retries=3) is True
        assert len(attempts) == 1

    def test_send_with_retry_eventual_success(self):
        ch = Channel("slack")
        attempts = []
        def op():
            attempts.append(1)
            return len(attempts) >= 3
        assert ch.send_with_retry(op, max_retries=5, base_delay=0.01) is True
        assert len(attempts) == 3

    def test_send_with_retry_all_fail(self):
        ch = Channel("slack")
        def op():
            raise RuntimeError("fail")
        assert ch.send_with_retry(op, max_retries=3, base_delay=0.01) is False
        stats = ch.get_stats()
        assert stats["errors"] == 3


class TestChannelManager:
    def test_creation(self):
        mgr = ChannelManager()
        assert mgr.channel_count() == 0

    def test_register(self):
        mgr = ChannelManager()
        ch = Channel("slack")
        mgr.register(ch)
        assert mgr.channel_count() == 1
        assert "slack" in mgr.list_channels()

    def test_register_with_policy(self):
        mgr = ChannelManager()
        ch = Channel("github")
        policy = ChannelRunPolicy(is_interactive=False, fire_and_forget=True)
        mgr.register(ch, policy)
        assert mgr.get_policy("github") == policy

    def test_unregister(self):
        mgr = ChannelManager()
        mgr.register(Channel("slack"))
        mgr.unregister("slack")
        assert mgr.channel_count() == 0

    def test_unregister_stops_running(self):
        mgr = ChannelManager()
        ch = Channel("slack")
        ch.start()
        mgr.register(ch)
        mgr.unregister("slack")
        assert ch.is_running is False

    def test_get(self):
        mgr = ChannelManager()
        ch = Channel("slack")
        mgr.register(ch)
        assert mgr.get("slack") is ch

    def test_get_nonexistent(self):
        mgr = ChannelManager()
        assert mgr.get("nonexistent") is None

    def test_get_policy_default(self):
        mgr = ChannelManager()
        ch = Channel("slack")
        mgr.register(ch)
        policy = mgr.get_policy("slack")
        assert policy is not None
        assert policy.is_interactive is True

    def test_get_policy_nonexistent(self):
        mgr = ChannelManager()
        assert mgr.get_policy("nonexistent") is None

    def test_list_channels(self):
        mgr = ChannelManager()
        mgr.register(Channel("slack"))
        mgr.register(Channel("discord"))
        channels = mgr.list_channels()
        assert "slack" in channels
        assert "discord" in channels

    def test_start_all(self):
        mgr = ChannelManager()
        ch1 = Channel("slack")
        ch2 = Channel("discord")
        mgr.register(ch1)
        mgr.register(ch2)
        mgr.start_all()
        assert ch1.is_running is True
        assert ch2.is_running is True

    def test_stop_all(self):
        mgr = ChannelManager()
        ch1 = Channel("slack")
        ch2 = Channel("discord")
        mgr.register(ch1)
        mgr.register(ch2)
        mgr.start_all()
        mgr.stop_all()
        assert ch1.is_running is False
        assert ch2.is_running is False

    def test_send(self):
        mgr = ChannelManager()
        mgr.register(Channel("slack"))
        msg = OutboundMessage("slack", "C1", "reply")
        # Default Channel.send returns False
        assert mgr.send(msg) is False

    def test_send_unknown_channel(self):
        mgr = ChannelManager()
        msg = OutboundMessage("nonexistent", "C1", "reply")
        assert mgr.send(msg) is False

    def test_all_stats(self):
        mgr = ChannelManager()
        ch1 = Channel("slack")
        ch2 = Channel("discord")
        mgr.register(ch1)
        mgr.register(ch2)
        ch1.send(OutboundMessage("slack", "C1", "msg"))
        stats = mgr.all_stats()
        assert "slack" in stats
        assert "discord" in stats
        assert stats["slack"]["outbound_count"] == 1


class TestCustomChannel:
    def test_subclass_send(self):
        class SlackChannel(Channel):
            def __init__(self):
                super().__init__("slack")
                self.sent = []

            def send(self, msg):
                self.sent.append(msg.content)
                return True

        ch = SlackChannel()
        ch.send(OutboundMessage("slack", "C1", "hello"))
        assert ch.sent == ["hello"]

    def test_subclass_supports_streaming(self):
        class StreamingChannel(Channel):
            @property
            def supports_streaming(self):
                return True

        ch = StreamingChannel("stream")
        assert ch.supports_streaming is True
