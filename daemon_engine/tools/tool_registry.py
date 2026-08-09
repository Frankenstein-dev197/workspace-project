"""Tool Registry: unified registry and dispatch for all agent tools.

Inspired by DeerFlow's tool search/builtins system and LangChain's BaseTool.
Provides registration, discovery, and execution with safety checks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: str = ""
    error: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    duration: float = 0.0

    def __str__(self) -> str:
        if self.success:
            return self.output
        return f"Error: {self.error}"


@dataclass
class ToolDefinition:
    name: str
    description: str
    category: str
    handler: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    is_safe: bool = True
    requires_confirmation: bool = False


class ToolRegistry:
    """Central registry for all tools available to agents."""

    DANGEROUS_PATTERNS = [
        "rm -rf /", "sudo ", "shutdown", "reboot", "mkfs",
        "dd if=/dev/zero", "> /dev/sda", ":(){:|:&};:",
    ]

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self._execution_log: list[dict[str, Any]] = []

    def register(
        self,
        name: str,
        description: str,
        handler: Callable[..., Any],
        category: str = "general",
        parameters: dict[str, Any] | None = None,
        is_safe: bool = True,
        requires_confirmation: bool = False,
    ) -> None:
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            handler=handler,
            category=category,
            parameters=parameters or {},
            is_safe=is_safe,
            requires_confirmation=requires_confirmation,
        )
        logger.info("Registered tool: %s (%s)", name, category)

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def list_by_category(self, category: str) -> list[str]:
        return [name for name, tool in self._tools.items() if tool.category == category]

    def get_tool(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def get_descriptions(self) -> dict[str, str]:
        return {name: tool.description for name, tool in self._tools.items()}

    def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(tool_name=tool_name, success=False, error=f"Tool '{tool_name}' not found")
        for key, val in kwargs.items():
            if isinstance(val, str) and any(p in val for p in self.DANGEROUS_PATTERNS):
                logger.warning("Blocked dangerous input in tool %s: %s", tool_name, val[:50])
                return ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=f"Dangerous pattern detected in input",
                )
        start = time.time()
        try:
            result = tool.handler(**kwargs)
            if isinstance(result, ToolResult):
                result.duration = time.time() - start
                return result
            output = str(result) if result is not None else "(no output)"
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=output[:10000],
                duration=time.time() - start,
            )
        except Exception as exc:
            logger.error("Tool %s execution error: %s", tool_name, exc)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(exc),
                duration=time.time() - start,
            )

    def get_execution_log(self) -> list[dict[str, Any]]:
        return self._execution_log

    def clear_log(self) -> None:
        self._execution_log.clear()
