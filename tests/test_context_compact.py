"""Tests for context compaction pipeline."""

import pytest
from pathlib import Path

from daemon_engine.core.context_compact import (
    ContextCompactor,
    snip_compact,
    micro_compact,
    tool_result_budget,
    persist_large_output,
    compact_history,
    reactive_compact,
    auto_compact,
    estimate_tokens,
    needs_compaction,
    summarize_history_static,
    write_transcript,
    MAX_MESSAGES_THRESHOLD,
    KEEP_RECENT_TOOL_RESULTS,
    PERSIST_THRESHOLD,
)


def make_messages(n: int) -> list[dict]:
    return [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"} for i in range(n)]


def make_tool_result_messages(n: int) -> list[dict]:
    messages = []
    for i in range(n):
        messages.append({"role": "assistant", "content": [{"type": "tool_use", "name": "bash", "id": f"call-{i}", "input": {}}]})
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": f"call-{i}", "content": f"Output {i} " * 50}]})
    return messages


class TestSnipCompact:
    def test_no_compaction_needed(self):
        messages = make_messages(10)
        result = snip_compact(messages, max_messages=50)
        assert len(result) == 10

    def test_compaction_triggered(self):
        messages = make_messages(60)
        result = snip_compact(messages, max_messages=20)
        assert len(result) < 60
        assert "snipped" in str(result)

    def test_keeps_head_and_tail(self):
        messages = make_messages(60)
        result = snip_compact(messages, max_messages=20)
        assert result[0]["content"] == "Message 0"
        assert result[-1]["content"] == "Message 59"

    def test_handles_tool_result_pairs(self):
        messages = make_tool_result_messages(30)
        result = snip_compact(messages, max_messages=10)
        assert len(result) < len(messages)


class TestMicroCompact:
    def test_no_compaction_needed(self):
        messages = make_tool_result_messages(3)
        result = micro_compact(messages, keep_recent=6)
        assert len(result) == len(messages)

    def test_compacts_old_results(self):
        messages = make_tool_result_messages(10)
        result = micro_compact(messages, keep_recent=3)
        tool_results = [b for msg in result for b in (msg.get("content", []) if isinstance(msg.get("content"), list) else []) if isinstance(b, dict) and b.get("type") == "tool_result"]
        compacted = [t for t in tool_results if "compacted" in str(t.get("content", ""))]
        assert len(compacted) > 0

    def test_keeps_recent_results(self):
        messages = make_tool_result_messages(10)
        result = micro_compact(messages, keep_recent=3)
        tool_results = [b for msg in result for b in (msg.get("content", []) if isinstance(msg.get("content"), list) else []) if isinstance(b, dict) and b.get("type") == "tool_result"]
        recent = tool_results[-3:]
        for tr in recent:
            assert "compacted" not in str(tr.get("content", ""))


class TestPersistLargeOutput:
    def test_small_output_not_persisted(self, tmp_path):
        output = "small output"
        result = persist_large_output("call-1", output, tmp_path)
        assert result == output

    def test_large_output_persisted(self, tmp_path):
        output = "x" * 6000
        result = persist_large_output("call-1", output, tmp_path)
        assert "persisted-output" in result
        assert (tmp_path / "call-1.txt").exists()

    def test_persisted_file_content(self, tmp_path):
        output = "x" * 6000
        persist_large_output("call-1", output, tmp_path)
        saved = (tmp_path / "call-1.txt").read_text()
        assert saved == output


class TestToolResultBudget:
    def test_no_budget_exceeded(self):
        messages = [{"role": "user", "content": [{"type": "tool_result", "content": "small"}]}]
        result = tool_result_budget(messages, max_bytes=200000)
        assert result == messages

    def test_budget_exceeded(self, tmp_path):
        large_content = "x" * 100000
        messages = [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call-1", "content": large_content},
            {"type": "tool_result", "tool_use_id": "call-2", "content": large_content},
        ]}]
        result = tool_result_budget(messages, max_bytes=50000, persist_dir=tmp_path)
        total = sum(len(str(b.get("content", ""))) for b in result[-1]["content"] if isinstance(b, dict) and b.get("type") == "tool_result")
        assert total < 200000

    def test_non_list_content_skipped(self):
        messages = [{"role": "user", "content": "text content"}]
        result = tool_result_budget(messages)
        assert result == messages


class TestCompactHistory:
    def test_compact_with_static_summary(self):
        messages = make_messages(20)
        result = compact_history(messages)
        assert len(result) == 1
        assert "Compacted" in result[0]["content"]

    def test_compact_with_custom_summarizer(self):
        messages = make_messages(10)
        result = compact_history(messages, summarize_fn=lambda msgs: "Custom summary")
        assert "Custom summary" in result[0]["content"]

    def test_transcript_written(self, tmp_path):
        messages = make_messages(5)
        compact_history(messages, transcript_dir=tmp_path)
        transcripts = list(tmp_path.glob("transcript_*.jsonl"))
        assert len(transcripts) == 1


class TestReactiveCompact:
    def test_reactive_keeps_tail(self):
        messages = make_messages(20)
        result = reactive_compact(messages, keep_tail=5)
        assert len(result) <= 6
        assert "Reactive compact" in result[0]["content"]

    def test_reactive_with_custom_summarizer(self):
        messages = make_messages(10)
        result = reactive_compact(messages, summarize_fn=lambda msgs: "Emergency summary")
        assert "Emergency summary" in result[0]["content"]


class TestEstimateTokens:
    def test_estimate(self):
        messages = [{"role": "user", "content": "Hello world"}]
        tokens = estimate_tokens(messages)
        assert tokens > 0

    def test_empty_messages(self):
        assert estimate_tokens([]) == 0

    def test_list_content(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        tokens = estimate_tokens(messages)
        assert tokens > 0


class TestNeedsCompaction:
    def test_no_compaction_needed(self):
        messages = make_messages(10)
        assert needs_compaction(messages, max_messages=50) is None

    def test_snip_needed(self):
        messages = make_messages(60)
        assert needs_compaction(messages, max_messages=50) == "snip"

    def test_full_needed(self):
        messages = [{"role": "user", "content": "x" * 500000}]
        result = needs_compaction(messages, max_tokens=1000)
        assert result == "full"


class TestAutoCompact:
    def test_no_compaction_needed(self):
        messages = make_messages(10)
        result, level = auto_compact(messages, max_messages=50)
        assert level == "none"
        assert len(result) == len(messages)

    def test_snip_triggered(self):
        messages = make_messages(60)
        result, level = auto_compact(messages, max_messages=20)
        assert level == "snip"
        assert len(result) < 60

    def test_with_tool_results(self):
        messages = make_tool_result_messages(20)
        result, level = auto_compact(messages, max_messages=10)
        assert level in ("snip", "full", "none")


class TestContextCompactor:
    def test_creation(self):
        compactor = ContextCompactor()
        assert compactor.max_messages == MAX_MESSAGES_THRESHOLD

    def test_compact(self):
        compactor = ContextCompactor(max_messages=10)
        messages = make_messages(20)
        result, level = compactor.compact(messages)
        assert level in ("snip", "full")
        assert len(result) < 20

    def test_reactive(self):
        compactor = ContextCompactor()
        messages = make_messages(20)
        result = compactor.reactive(messages)
        assert len(result) < 20

    def test_check_needed(self):
        compactor = ContextCompactor(max_messages=10)
        messages = make_messages(5)
        assert compactor.check_needed(messages) is None
        messages = make_messages(20)
        assert compactor.check_needed(messages) is not None

    def test_stats_dict(self):
        compactor = ContextCompactor(max_messages=10)
        messages = make_messages(20)
        compactor.compact(messages)
        stats = compactor.stats_dict()
        assert "snip_compactions" in stats
        assert "max_messages" in stats


class TestSummarizeHistory:
    def test_static_summary(self):
        messages = make_messages(5)
        summary = summarize_history_static(messages)
        assert "Message 0" in summary
        assert "Message 4" in summary

    def test_truncation(self):
        messages = [{"role": "user", "content": "x" * 200}]
        summary = summarize_history_static(messages, max_chars=50)
        assert len(summary) <= 100
