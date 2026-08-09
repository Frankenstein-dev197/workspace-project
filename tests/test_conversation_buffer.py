"""Tests for conversation summary buffer."""

import pytest

from daemon_engine.memory.conversation_buffer import (
    ConversationSummaryBuffer,
    ConversationMessage,
    MessageRole,
    estimate_tokens,
    estimate_messages_tokens,
)


class TestMessageRole:
    def test_values(self):
        assert MessageRole.HUMAN.value == "human"
        assert MessageRole.AI.value == "ai"
        assert MessageRole.SYSTEM.value == "system"


class TestConversationMessage:
    def test_creation(self):
        msg = ConversationMessage(role=MessageRole.HUMAN, content="hello")
        assert msg.role == MessageRole.HUMAN
        assert msg.content == "hello"
        assert msg.timestamp > 0

    def test_to_dict(self):
        msg = ConversationMessage(role=MessageRole.AI, content="hi")
        d = msg.to_dict()
        assert d["role"] == "ai"
        assert d["content"] == "hi"

    def test_from_dict(self):
        d = {"role": "human", "content": "test", "timestamp": 123.0}
        msg = ConversationMessage.from_dict(d)
        assert msg.role == MessageRole.HUMAN
        assert msg.content == "test"
        assert msg.timestamp == 123.0

    def test_from_dict_default_role(self):
        d = {"content": "test"}
        msg = ConversationMessage.from_dict(d)
        assert msg.role == MessageRole.HUMAN


class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("hello world") > 0

    def test_empty(self):
        # Empty string still returns at least 1
        assert estimate_tokens("") == 1

    def test_long_text(self):
        tokens = estimate_tokens("a" * 100)
        assert tokens == 25  # 100 // 4

    def test_custom_chars_per_token(self):
        assert estimate_tokens("aaaa", chars_per_token=2) == 2


class TestEstimateMessagesTokens:
    def test_empty(self):
        assert estimate_messages_tokens([]) == 0

    def test_single_message(self):
        msg = ConversationMessage(MessageRole.HUMAN, "hello")
        tokens = estimate_messages_tokens([msg])
        # 5 chars // 4 = 1 token + 4 overhead = 5
        assert tokens == 5

    def test_multiple_messages(self):
        msgs = [
            ConversationMessage(MessageRole.HUMAN, "hello"),
            ConversationMessage(MessageRole.AI, "world"),
        ]
        tokens = estimate_messages_tokens(msgs)
        assert tokens > 0


class TestConversationSummaryBuffer:
    def test_creation(self):
        buf = ConversationSummaryBuffer(max_token_limit=2000)
        assert buf.max_token_limit == 2000
        assert buf.message_count == 0
        assert buf.moving_summary == ""

    def test_memory_variables(self):
        buf = ConversationSummaryBuffer()
        assert buf.memory_variables == ["history"]

    def test_save_context(self):
        buf = ConversationSummaryBuffer(max_token_limit=2000)
        buf.save_context("hello", "hi there")
        assert buf.message_count == 2

    def test_add_message(self):
        buf = ConversationSummaryBuffer()
        msg = ConversationMessage(MessageRole.SYSTEM, "system msg")
        buf.add_message(msg)
        assert buf.message_count == 1

    def test_buffer_string(self):
        buf = ConversationSummaryBuffer()
        buf.save_context("hello", "hi")
        buffer = buf.buffer
        assert "Human: hello" in buffer
        assert "Ai: hi" in buffer

    def test_load_memory_variables_string(self):
        buf = ConversationSummaryBuffer()
        buf.save_context("hello", "hi")
        result = buf.load_memory_variables()
        assert "history" in result
        assert "Human: hello" in result["history"]

    def test_load_memory_variables_messages(self):
        buf = ConversationSummaryBuffer()
        buf.save_context("hello", "hi")
        result = buf.load_memory_variables(return_messages=True)
        msgs = result["history"]
        assert len(msgs) == 2
        assert msgs[0].content == "hello"

    def test_clear(self):
        buf = ConversationSummaryBuffer()
        buf.save_context("hello", "hi")
        buf.clear()
        assert buf.message_count == 0
        assert buf.moving_summary == ""

    def test_token_count_empty(self):
        buf = ConversationSummaryBuffer()
        assert buf.token_count == 0

    def test_token_count_with_messages(self):
        buf = ConversationSummaryBuffer()
        buf.save_context("hello world", "hi there")
        assert buf.token_count > 0


class TestPruning:
    def test_no_prune_under_limit(self):
        buf = ConversationSummaryBuffer(max_token_limit=10000)
        buf.save_context("hello", "hi")
        assert buf.message_count == 2
        assert buf.moving_summary == ""

    def test_prune_when_over_limit(self):
        buf = ConversationSummaryBuffer(max_token_limit=15)
        # Each message ~10 tokens (6 content + 4 overhead)
        # 15 token limit → pruning needed
        buf.save_context("hello world this is long", "yes it certainly is long")
        assert buf.moving_summary != ""
        # Pruning reduces buffer messages (not summary) to fit limit
        assert len(buf.messages) <= 2

    def test_prune_creates_summary(self):
        buf = ConversationSummaryBuffer(max_token_limit=15)
        buf.save_context("first message here", "first response here")
        buf.save_context("second message here", "second response here")
        assert buf.moving_summary != ""
        assert "first" in buf.moving_summary

    def test_prune_keeps_recent(self):
        buf = ConversationSummaryBuffer(max_token_limit=20)
        buf.save_context("old message one", "old response one")
        buf.save_context("new message two", "new response two")
        # Recent messages should still be in buffer
        msgs = buf.messages
        contents = [m.content for m in msgs]
        assert any("new" in c for c in contents)

    def test_explicit_prune(self):
        buf = ConversationSummaryBuffer(max_token_limit=10000)
        buf.save_context("hello", "hi")
        buf.prune()  # should be no-op when under limit
        assert buf.message_count == 2

    def test_summary_in_load_variables(self):
        buf = ConversationSummaryBuffer(max_token_limit=15)
        buf.save_context("first message", "first response")
        buf.save_context("second message", "second response")
        result = buf.load_memory_variables()
        assert "Summary:" in result["history"]


class TestCustomSummarizer:
    def test_custom_summarizer(self):
        def my_summarizer(messages, existing):
            return f"[{len(messages)} pruned] {existing}"

        buf = ConversationSummaryBuffer(
            max_token_limit=15,
            summarizer=my_summarizer,
        )
        buf.save_context("first message here", "first response here")
        buf.save_context("second message here", "second response here")
        assert buf.moving_summary.startswith("[")
        assert "pruned" in buf.moving_summary


class TestSerialization:
    def test_to_dict(self):
        buf = ConversationSummaryBuffer(max_token_limit=1000)
        buf.save_context("hello", "hi")
        d = buf.to_dict()
        assert d["max_token_limit"] == 1000
        assert len(d["messages"]) == 2
        assert d["moving_summary"] == ""

    def test_from_dict(self):
        buf1 = ConversationSummaryBuffer(max_token_limit=1000)
        buf1.save_context("hello", "hi")
        d = buf1.to_dict()

        buf2 = ConversationSummaryBuffer.from_dict(d)
        assert buf2.max_token_limit == 1000
        assert buf2.message_count == 2

    def test_from_dict_with_summary(self):
        d = {
            "max_token_limit": 500,
            "memory_key": "history",
            "moving_summary": "Previous conversation summary",
            "messages": [
                {"role": "human", "content": "recent", "timestamp": 1.0},
                {"role": "ai", "content": "reply", "timestamp": 2.0},
            ],
        }
        buf = ConversationSummaryBuffer.from_dict(d)
        assert buf.moving_summary == "Previous conversation summary"
        assert buf.message_count == 2

    def test_round_trip(self):
        buf1 = ConversationSummaryBuffer(max_token_limit=100)
        buf1.save_context("message one", "response one")
        buf1.save_context("message two", "response two")
        d = buf1.to_dict()
        buf2 = ConversationSummaryBuffer.from_dict(d)
        assert buf2.moving_summary == buf1.moving_summary
        assert buf2.message_count == buf1.message_count


class TestThreadSafety:
    def test_concurrent_save_context(self):
        import threading

        buf = ConversationSummaryBuffer(max_token_limit=10000)
        errors = []

        def writer(i):
            try:
                for j in range(10):
                    buf.save_context(f"msg-{i}-{j}", f"resp-{i}-{j}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert buf.message_count == 100
