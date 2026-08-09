"""Tests for subagent system."""

import pytest

from daemon_engine.multi_agent.subagent import (
    Subagent,
    SubagentConfig,
    SubagentManager,
    SubagentResult,
    SubagentStatus,
)
from daemon_engine.core.hooks import HookRegistry


class TestSubagentConfig:
    def test_defaults(self):
        config = SubagentConfig(name="test", description="A test subagent")
        assert config.name == "test"
        assert config.max_turns == 50
        assert config.timeout_seconds == 900
        assert "task" in config.disallowed_tools

    def test_custom_config(self):
        config = SubagentConfig(
            name="coder",
            description="Code subagent",
            tools=["read_file", "write_file"],
            max_turns=100,
            model="gpt-4",
        )
        assert config.tools == ["read_file", "write_file"]
        assert config.max_turns == 100
        assert config.model == "gpt-4"


class TestSubagent:
    def test_creation(self):
        config = SubagentConfig(name="test", description="Test")
        sub = Subagent(config=config)
        assert sub.status == SubagentStatus.PENDING
        assert sub.id.startswith("subagent-")

    def test_model_inheritance(self):
        config = SubagentConfig(name="test", description="Test", model="inherit")
        sub = Subagent(config=config, parent_model="gpt-4")
        assert sub.model == "gpt-4"

    def test_model_override(self):
        config = SubagentConfig(name="test", description="Test", model="claude-3")
        sub = Subagent(config=config, parent_model="gpt-4")
        assert sub.model == "claude-3"

    def test_execute_no_llm(self):
        config = SubagentConfig(name="test", description="Test", max_turns=1)
        sub = Subagent(config=config)
        result = sub.execute("Do something")
        assert result.status == SubagentStatus.COMPLETED
        assert result.subagent_id == sub.id
        assert result.turns_used >= 1

    def test_execute_with_callback(self):
        config = SubagentConfig(name="test", description="Test", max_turns=5)
        sub = Subagent(config=config)
        call_count = [0]

        def callback(messages, tools, model):
            call_count[0] += 1
            return {"content": f"Response {call_count[0]}", "tool_calls": []}

        result = sub.execute("Test task", llm_callback=callback)
        assert result.status == SubagentStatus.COMPLETED
        assert call_count[0] >= 1

    def test_execute_with_tool_calls(self):
        config = SubagentConfig(name="test", description="Test", max_turns=5)

        def callback(messages, tools, model):
            if len(messages) <= 2:
                return {
                    "content": "Using a tool",
                    "tool_calls": [{"name": "read_file", "input": {"path": "test.py"}, "id": "call-1"}],
                }
            return {"content": "Done", "tool_calls": []}

        sub = Subagent(config=config)
        result = sub.execute("Read a file", llm_callback=callback)
        assert result.status == SubagentStatus.COMPLETED
        assert "read_file" in result.tools_used

    def test_max_turns_limit(self):
        config = SubagentConfig(name="test", description="Test", max_turns=3)

        def callback(messages, tools, model):
            return {"content": "Continuing", "tool_calls": [{"name": "tool", "input": {}, "id": "1"}]}

        sub = Subagent(config=config)
        result = sub.execute("Keep going", llm_callback=callback)
        assert result.turns_used <= 3

    def test_cancel(self):
        config = SubagentConfig(name="test", description="Test")
        sub = Subagent(config=config)
        sub.cancel()
        assert sub.status == SubagentStatus.CANCELLED

    def test_stats(self):
        config = SubagentConfig(name="test", description="Test")
        sub = Subagent(config=config)
        stats = sub.stats()
        assert stats["name"] == "test"
        assert stats["status"] == "pending"

    def test_hooks_integration(self):
        hooks = HookRegistry()
        config = SubagentConfig(name="test", description="Test", max_turns=1)
        sub = Subagent(config=config, hooks=hooks)
        sub.execute("Test task")
        assert sub.status == SubagentStatus.COMPLETED


class TestSubagentManager:
    def test_default_configs(self):
        mgr = SubagentManager()
        configs = mgr.list_configs()
        names = [c["name"] for c in configs]
        assert "general-purpose" in names
        assert "coder" in names
        assert "researcher" in names
        assert "bash" in names

    def test_register_config(self):
        mgr = SubagentManager()
        config = SubagentConfig(name="custom", description="Custom subagent")
        mgr.register_config(config)
        assert mgr.get_config("custom") is not None

    def test_spawn(self):
        mgr = SubagentManager()
        result = mgr.spawn("Do a task", config_name="general-purpose")
        assert result.status in (SubagentStatus.COMPLETED, SubagentStatus.FAILED)
        assert result.subagent_id.startswith("subagent-")

    def test_spawn_unknown_config(self):
        mgr = SubagentManager()
        result = mgr.spawn("Do a task", config_name="nonexistent")
        assert result.status in (SubagentStatus.COMPLETED, SubagentStatus.FAILED)

    def test_spawn_with_callback(self):
        mgr = SubagentManager()

        def callback(messages, tools, model):
            return {"content": "Done", "tool_calls": []}

        result = mgr.spawn("Test", llm_callback=callback)
        assert result.status == SubagentStatus.COMPLETED

    def test_get_completed(self):
        mgr = SubagentManager()
        mgr.spawn("Task 1")
        mgr.spawn("Task 2")
        completed = mgr.get_completed()
        assert len(completed) >= 2

    def test_stats(self):
        mgr = SubagentManager()
        mgr.spawn("Task 1")
        stats = mgr.stats()
        assert stats["registered_configs"] >= 4
        assert stats["completed_count"] >= 1

    def test_recursion_prevention(self):
        """Subagents should not have 'task' tool (prevents recursion)."""
        mgr = SubagentManager()
        configs = mgr.list_configs()
        for config in configs:
            assert "task" in config["disallowed_tools"]
