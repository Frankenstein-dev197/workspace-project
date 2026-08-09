"""Agent hooks system: extension points for the agent loop.

Integrates learn-claude-code's hooks pattern (s04_hooks): hooks are
registered callbacks triggered at specific points in the agent cycle,
keeping the core loop clean and extensible.

Hook events:
- UserPromptSubmit: before LLM call (input validation, context injection)
- PreToolUse: before tool execution (permission checks, logging)
- PostToolUse: after tool execution (side effects, output checks)
- Stop: when loop is about to exit (cleanup, summary)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class HookEvent(str, Enum):
    """Agent lifecycle hook events."""
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    STOP = "Stop"
    AGENT_START = "AgentStart"
    AGENT_END = "AgentEnd"
    ERROR = "Error"


@dataclass
class HookContext:
    """Context passed to hook callbacks."""
    event: HookEvent
    tool_name: str = ""
    tool_input: dict[str, Any] = field(default_factory=dict)
    tool_output: str = ""
    agent_id: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


HookCallback = Callable[[HookContext], Any]


@dataclass
class HookResult:
    """Result from a hook callback. Non-None value blocks or modifies behavior."""
    blocked: bool = False
    block_reason: str = ""
    modified_input: dict[str, Any] | None = None
    modified_output: str | None = None
    continue_loop: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


class HookRegistry:
    """Registry for hook callbacks, inspired by learn-claude-code s04.

    Hooks are registered per event and triggered in registration order.
    A hook returning a block result stops further processing for that event.
    """

    def __init__(self) -> None:
        self._hooks: dict[HookEvent, list[HookCallback]] = {
            event: [] for event in HookEvent
        }
        self._enabled: bool = True
        self._hook_count: int = 0

    def register(self, event: HookEvent, callback: HookCallback) -> None:
        if callback not in self._hooks[event]:
            self._hooks[event].append(callback)
            self._hook_count += 1
            logger.debug("Registered hook for %s: %s", event.value, callback.__name__)

    def unregister(self, event: HookEvent, callback: HookCallback) -> None:
        if callback in self._hooks[event]:
            self._hooks[event].remove(callback)

    def trigger(self, event: HookEvent, context: HookContext) -> HookResult | None:
        if not self._enabled:
            return None
        for callback in self._hooks.get(event, []):
            try:
                result = callback(context)
                if result is not None:
                    if isinstance(result, HookResult):
                        if result.blocked:
                            logger.info("Hook %s blocked %s: %s", callback.__name__, event.value, result.block_reason)
                            return result
                        if result.modified_input or result.modified_output:
                            return result
                        if not result.continue_loop:
                            return result
                    elif isinstance(result, str):
                        return HookResult(blocked=True, block_reason=result)
                    elif isinstance(result, bool):
                        if not result:
                            return HookResult(blocked=True, block_reason="Hook returned False")
            except Exception as exc:
                logger.error("Hook %s error on %s: %s", callback.__name__, event.value, exc)
        return None

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def list_hooks(self) -> dict[str, list[str]]:
        return {
            event.value: [cb.__name__ for cb in callbacks]
            for event, callbacks in self._hooks.items()
            if callbacks
        }

    def clear(self) -> None:
        for event in self._hooks:
            self._hooks[event].clear()
        self._hook_count = 0

    def stats(self) -> dict[str, Any]:
        return {
            "total_hooks": self._hook_count,
            "enabled": self._enabled,
            "by_event": {
                event.value: len(callbacks)
                for event, callbacks in self._hooks.items()
            },
        }


def permission_hook(context: HookContext) -> HookResult | None:
    """Default permission hook: block dangerous tool calls."""
    dangerous_commands = ["rm -rf /", "sudo ", "shutdown", "mkfs", "dd if=/dev/zero"]
    if context.tool_name in ("bash", "execute_command", "shell"):
        command = context.tool_input.get("command", "")
        for dangerous in dangerous_commands:
            if dangerous in command:
                return HookResult(
                    blocked=True,
                    block_reason=f"Blocked dangerous command pattern: {dangerous}",
                )
    return None


def logging_hook(context: HookContext) -> None:
    """Default logging hook: record tool calls."""
    logger.info(
        "[Hook] %s called %s with %s",
        context.agent_id or "agent",
        context.tool_name,
        list(context.tool_input.keys()),
    )


def large_output_hook(context: HookContext) -> HookResult | None:
    """PostToolUse hook: truncate large outputs."""
    if context.tool_output and len(context.tool_output) > 50000:
        return HookResult(
            modified_output=context.tool_output[:50000] + "\n... [truncated by hook]",
        )
    return None


def create_default_registry() -> HookRegistry:
    """Create a registry with default hooks pre-installed."""
    registry = HookRegistry()
    registry.register(HookEvent.PRE_TOOL_USE, permission_hook)
    registry.register(HookEvent.PRE_TOOL_USE, logging_hook)
    registry.register(HookEvent.POST_TOOL_USE, large_output_hook)
    return registry
