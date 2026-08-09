"""Tests for hooks system and message manager."""

import pytest

from daemon_engine.core.hooks import (
    HookRegistry,
    HookEvent,
    HookContext,
    HookResult,
    create_default_registry,
    permission_hook,
    logging_hook,
    large_output_hook,
)
from daemon_engine.core.message_manager import (
    MessageManager,
    Message,
    MessageRole,
    CompactionSettings,
)


class TestHookRegistry:
    def test_register_and_trigger(self):
        registry = HookRegistry()
        called = []

        def my_hook(ctx):
            called.append(ctx.tool_name)
            return None

        registry.register(HookEvent.PRE_TOOL_USE, my_hook)
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="bash")
        registry.trigger(HookEvent.PRE_TOOL_USE, ctx)
        assert called == ["bash"]

    def test_block_result(self):
        registry = HookRegistry()

        def blocking_hook(ctx):
            return HookResult(blocked=True, block_reason="Not allowed")

        registry.register(HookEvent.PRE_TOOL_USE, blocking_hook)
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE, tool_name="bash")
        result = registry.trigger(HookEvent.PRE_TOOL_USE, ctx)
        assert result is not None
        assert result.blocked is True

    def test_multiple_hooks(self):
        registry = HookRegistry()
        order = []

        def hook1(ctx):
            order.append(1)

        def hook2(ctx):
            order.append(2)

        registry.register(HookEvent.PRE_TOOL_USE, hook1)
        registry.register(HookEvent.PRE_TOOL_USE, hook2)
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE)
        registry.trigger(HookEvent.PRE_TOOL_USE, ctx)
        assert order == [1, 2]

    def test_unregister(self):
        registry = HookRegistry()
        called = []

        def my_hook(ctx):
            called.append(1)

        registry.register(HookEvent.PRE_TOOL_USE, my_hook)
        registry.unregister(HookEvent.PRE_TOOL_USE, my_hook)
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE)
        registry.trigger(HookEvent.PRE_TOOL_USE, ctx)
        assert called == []

    def test_enable_disable(self):
        registry = HookRegistry()
        called = []

        def my_hook(ctx):
            called.append(1)

        registry.register(HookEvent.PRE_TOOL_USE, my_hook)
        registry.disable()
        ctx = HookContext(event=HookEvent.PRE_TOOL_USE)
        registry.trigger(HookEvent.PRE_TOOL_USE, ctx)
        assert called == []
        registry.enable()
        registry.trigger(HookEvent.PRE_TOOL_USE, ctx)
        assert called == [1]

    def test_stats(self):
        registry = HookRegistry()
        registry.register(HookEvent.PRE_TOOL_USE, lambda ctx: None)
        stats = registry.stats()
        assert stats["total_hooks"] == 1
        assert stats["enabled"] is True

    def test_list_hooks(self):
        registry = HookRegistry()
        registry.register(HookEvent.PRE_TOOL_USE, permission_hook)
        hooks = registry.list_hooks()
        assert "PreToolUse" in hooks


class TestDefaultHooks:
    def test_permission_hook_blocks_dangerous(self):
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="bash",
            tool_input={"command": "rm -rf /"},
        )
        result = permission_hook(ctx)
        assert result is not None
        assert result.blocked is True

    def test_permission_hook_allows_safe(self):
        ctx = HookContext(
            event=HookEvent.PRE_TOOL_USE,
            tool_name="bash",
            tool_input={"command": "echo hello"},
        )
        result = permission_hook(ctx)
        assert result is None

    def test_large_output_hook(self):
        ctx = HookContext(
            event=HookEvent.POST_TOOL_USE,
            tool_output="x" * 60000,
        )
        result = large_output_hook(ctx)
        assert result is not None
        assert result.modified_output is not None
        assert "truncated" in result.modified_output

    def test_create_default_registry(self):
        registry = create_default_registry()
        stats = registry.stats()
        assert stats["total_hooks"] >= 3


class TestMessageManager:
    def test_add_message(self):
        mgr = MessageManager(system_prompt="You are an agent")
        mgr.add_user("Hello")
        mgr.add_assistant("Hi there")
        assert mgr.message_count == 3  # system + user + assistant

    def test_get_system_prompt(self):
        mgr = MessageManager(system_prompt="System prompt here")
        assert mgr.get_system_prompt() == "System prompt here"

    def test_get_recent(self):
        mgr = MessageManager()
        mgr.add_user("msg1")
        mgr.add_user("msg2")
        mgr.add_user("msg3")
        recent = mgr.get_recent(2)
        assert len(recent) == 2
        assert "msg2" in recent[0].content

    def test_clear(self):
        mgr = MessageManager(system_prompt="System")
        mgr.add_user("Hello")
        mgr.clear()
        assert mgr.message_count == 1  # only system remains

    def test_clear_all(self):
        mgr = MessageManager(system_prompt="System")
        mgr.add_user("Hello")
        mgr.clear(keep_system=False)
        assert mgr.message_count == 0

    def test_to_message_list(self):
        mgr = MessageManager()
        mgr.add_user("Hello")
        msgs = mgr.to_message_list()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"

    def test_add_tool_result(self):
        mgr = MessageManager()
        mgr.add_tool_result("call-1", "Tool output here")
        msgs = mgr.get_messages(MessageRole.TOOL)
        assert len(msgs) == 1
        assert msgs[0].tool_call_id == "call-1"

    def test_stats(self):
        mgr = MessageManager(system_prompt="System")
        mgr.add_user("Hello world")
        stats = mgr.stats()
        assert stats["message_count"] == 2
        assert stats["by_role"]["user"] == 1
        assert stats["by_role"]["system"] == 1

    def test_save_and_load_state(self):
        mgr = MessageManager(system_prompt="System")
        mgr.add_user("Hello")
        mgr.add_assistant("Hi")
        state = mgr.save_state()
        mgr2 = MessageManager()
        mgr2.load_state(state)
        assert mgr2.message_count == 3

    def test_compaction_disabled(self):
        settings = CompactionSettings(enabled=False)
        mgr = MessageManager(compaction=settings)
        for i in range(30):
            mgr.add_user(f"Message {i} " * 100)
        assert mgr.stats()["compaction_count"] == 0

    def test_estimated_tokens(self):
        mgr = MessageManager()
        mgr.add_user("Hello world this is a test")
        assert mgr.estimated_tokens > 0
