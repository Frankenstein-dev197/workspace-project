"""Subagent system: spawn isolated sub-agents for complex subtasks.

Integrates learn-claude-code's subagent pattern (s06): subagents get
fresh message context, a restricted tool set (no recursion), and return
only a summary. Intermediate results are discarded to maintain context
isolation.

Also integrates DeerFlow's SubagentConfig: model inheritance, max turns,
timeout, disallowed tools, and skill scoping.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from daemon_engine.core.hooks import HookContext, HookEvent, HookRegistry
from daemon_engine.core.message_manager import MessageManager, MessageRole

logger = logging.getLogger(__name__)


class SubagentStatus(str, Enum):
    """Subagent execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SubagentConfig:
    """Configuration for a subagent (from DeerFlow SubagentConfig).

    Attributes:
        name: Unique identifier for the subagent.
        description: When to delegate to this subagent.
        system_prompt: System prompt guiding subagent behavior.
        tools: Allowed tool names (None = inherit all).
        disallowed_tools: Tool names to deny (default: task to prevent recursion).
        skills: Skill names to make available (None = all, empty = none).
        model: Model to use ('inherit' uses parent's model).
        max_turns: Maximum turns before stopping.
        timeout_seconds: Execution time cap.
    """
    name: str
    description: str
    system_prompt: str | None = None
    tools: list[str] | None = None
    disallowed_tools: list[str] = field(default_factory=lambda: ["task"])
    skills: list[str] | None = None
    model: str = "inherit"
    max_turns: int = 50
    timeout_seconds: int = 900


@dataclass
class SubagentResult:
    """Result of a subagent execution."""
    subagent_id: str
    status: SubagentStatus
    summary: str = ""
    turns_used: int = 0
    duration_seconds: float = 0.0
    error: str = ""
    tools_used: list[str] = field(default_factory=list)
    tokens_estimated: int = 0

    @property
    def success(self) -> bool:
        return self.status == SubagentStatus.COMPLETED


class Subagent:
    """A single subagent instance with isolated context.

    Subagents have their own message manager, restricted tool set,
    and return only a summary to the parent agent.
    """

    def __init__(
        self,
        config: SubagentConfig,
        tool_registry: Any | None = None,
        hooks: HookRegistry | None = None,
        parent_model: str | None = None,
    ) -> None:
        self.config = config
        self.tool_registry = tool_registry
        self.hooks = hooks
        self.id = f"subagent-{uuid.uuid4().hex[:8]}"
        self.status = SubagentStatus.PENDING
        self._message_mgr = MessageManager(
            system_prompt=config.system_prompt or "You are a subagent. Complete the task and return a summary."
        )
        self._model = self._resolve_model(parent_model)
        self._tools_used: list[str] = []
        self._turn_count = 0
        self._start_time: float = 0.0
        self._result: SubagentResult | None = None

    def _resolve_model(self, parent_model: str | None) -> str:
        if self.config.model != "inherit":
            return self.config.model
        return parent_model or "default"

    @property
    def model(self) -> str:
        return self._model

    @property
    def message_count(self) -> int:
        return self._message_mgr.message_count

    def get_allowed_tools(self) -> list[str]:
        """Get the list of allowed tool names for this subagent."""
        if self.tool_registry is None:
            return []
        all_tools = self.tool_registry.list_tool_names() if hasattr(self.tool_registry, "list_tool_names") else []
        if self.config.tools is not None:
            allowed = [t for t in all_tools if t in self.config.tools]
        else:
            allowed = list(all_tools)
        allowed = [t for t in allowed if t not in self.config.disallowed_tools]
        return allowed

    def execute(self, task_description: str, llm_callback: Callable | None = None) -> SubagentResult:
        """Execute the subagent task.

        Args:
            task_description: The task to perform.
            llm_callback: Function(messages, tools, model) -> response.
                         If None, uses a simulated callback.

        Returns:
            SubagentResult with summary and metadata.
        """
        self.status = SubagentStatus.RUNNING
        self._start_time = time.time()
        self._message_mgr.add_user(task_description)
        self._trigger_hook(HookEvent.AGENT_START, tool_input={"task": task_description})

        try:
            while self._turn_count < self.config.max_turns:
                elapsed = time.time() - self._start_time
                if elapsed > self.config.timeout_seconds:
                    self.status = SubagentStatus.TIMEOUT
                    self._result = SubagentResult(
                        subagent_id=self.id,
                        status=SubagentStatus.TIMEOUT,
                        summary="Subagent timed out",
                        turns_used=self._turn_count,
                        duration_seconds=elapsed,
                        error=f"Exceeded {self.config.timeout_seconds}s timeout",
                        tools_used=self._tools_used,
                    )
                    return self._result

                self._turn_count += 1
                if llm_callback:
                    response = llm_callback(
                        self._message_mgr.to_message_list(),
                        self.get_allowed_tools(),
                        self._model,
                    )
                else:
                    response = self._simulate_response(task_description)

                self._message_mgr.add_assistant(str(response.get("content", "")))

                tool_calls = response.get("tool_calls", [])
                if not tool_calls:
                    break

                for tool_call in tool_calls:
                    tool_name = tool_call.get("name", "")
                    tool_input = tool_call.get("input", {})
                    self._tools_used.append(tool_name)
                    self._trigger_hook(
                        HookEvent.PRE_TOOL_USE,
                        tool_name=tool_name,
                        tool_input=tool_input,
                    )
                    output = self._execute_tool(tool_name, tool_input)
                    self._trigger_hook(
                        HookEvent.POST_TOOL_USE,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_output=output,
                    )
                    self._message_mgr.add_tool_result(
                        tool_call.get("id", str(self._turn_count)),
                        output,
                    )

            elapsed = time.time() - self._start_time
            summary = self._extract_summary()
            self.status = SubagentStatus.COMPLETED
            self._result = SubagentResult(
                subagent_id=self.id,
                status=SubagentStatus.COMPLETED,
                summary=summary,
                turns_used=self._turn_count,
                duration_seconds=elapsed,
                tools_used=list(set(self._tools_used)),
                tokens_estimated=self._message_mgr.estimated_tokens,
            )
        except Exception as exc:
            elapsed = time.time() - self._start_time
            logger.error("Subagent %s failed: %s", self.id, exc)
            self.status = SubagentStatus.FAILED
            self._result = SubagentResult(
                subagent_id=self.id,
                status=SubagentStatus.FAILED,
                summary="",
                turns_used=self._turn_count,
                duration_seconds=elapsed,
                error=str(exc),
                tools_used=list(set(self._tools_used)),
            )

        self._trigger_hook(HookEvent.AGENT_END, tool_output=self._result.summary if self._result else "")
        return self._result if self._result else SubagentResult(
            subagent_id=self.id,
            status=SubagentStatus.FAILED,
            error="No result produced",
        )

    def _execute_tool(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Execute a tool call within the subagent context."""
        if self.tool_registry and hasattr(self.tool_registry, "execute"):
            try:
                result = self.tool_registry.execute(tool_name, **tool_input)
                return str(result)
            except Exception as exc:
                return f"Tool error: {exc}"
        return f"Tool '{tool_name}' executed (simulated)"

    def _simulate_response(self, task: str) -> dict[str, Any]:
        """Simulate an LLM response for testing."""
        return {
            "content": f"Processing task: {task[:50]}...",
            "tool_calls": [],
        }

    def _extract_summary(self) -> str:
        """Extract the final summary from message history."""
        for msg in reversed(self._message_mgr.messages):
            if msg.role == MessageRole.ASSISTANT and msg.content.strip():
                return msg.content[:2000]
        return "Subagent completed without final summary."

    def _trigger_hook(
        self,
        event: HookEvent,
        tool_name: str = "",
        tool_input: dict[str, Any] | None = None,
        tool_output: str = "",
    ) -> None:
        if not self.hooks:
            return
        ctx = HookContext(
            event=event,
            tool_name=tool_name,
            tool_input=tool_input or {},
            tool_output=tool_output,
            agent_id=self.id,
            messages=self._message_mgr.to_message_list(),
        )
        self.hooks.trigger(event, ctx)

    def cancel(self) -> None:
        """Cancel the subagent."""
        self.status = SubagentStatus.CANCELLED

    def stats(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.config.name,
            "status": self.status.value,
            "model": self._model,
            "turns": self._turn_count,
            "max_turns": self.config.max_turns,
            "messages": self.message_count,
            "tools_available": len(self.get_allowed_tools()),
            "tools_used": list(set(self._tools_used)),
        }


class SubagentManager:
    """Manages subagent lifecycle and spawning.

    Maintains a registry of subagent configs and spawns instances on demand.
    Enforces recursion prevention (subagents cannot spawn subagents).
    """

    def __init__(
        self,
        tool_registry: Any | None = None,
        hooks: HookRegistry | None = None,
    ) -> None:
        self.tool_registry = tool_registry
        self.hooks = hooks
        self._configs: dict[str, SubagentConfig] = {}
        self._active: dict[str, Subagent] = {}
        self._completed: list[SubagentResult] = []
        self._register_default_configs()

    def _register_default_configs(self) -> None:
        defaults = [
            SubagentConfig(
                name="general-purpose",
                description="General-purpose subagent for any task",
                max_turns=150,
            ),
            SubagentConfig(
                name="coder",
                description="Subagent specialized in writing and editing code",
                tools=["read_file", "write_file", "edit_file", "execute_code", "execute_command"],
                max_turns=100,
            ),
            SubagentConfig(
                name="researcher",
                description="Subagent for research and information gathering",
                tools=["web_search", "web_fetch", "scrape_page"],
                max_turns=80,
            ),
            SubagentConfig(
                name="bash",
                description="Subagent for running shell commands",
                tools=["execute_command"],
                max_turns=60,
            ),
        ]
        for config in defaults:
            self._configs[config.name] = config

    def register_config(self, config: SubagentConfig) -> None:
        self._configs[config.name] = config

    def get_config(self, name: str) -> SubagentConfig | None:
        return self._configs.get(name)

    def list_configs(self) -> list[dict[str, Any]]:
        return [
            {
                "name": c.name,
                "description": c.description,
                "max_turns": c.max_turns,
                "model": c.model,
                "tools": c.tools,
                "disallowed_tools": c.disallowed_tools,
            }
            for c in self._configs.values()
        ]

    def spawn(
        self,
        task_description: str,
        config_name: str = "general-purpose",
        parent_model: str | None = None,
        llm_callback: Callable | None = None,
    ) -> SubagentResult:
        """Spawn a subagent to handle a task."""
        config = self._configs.get(config_name)
        if not config:
            config = SubagentConfig(
                name=config_name,
                description=f"Custom subagent: {config_name}",
            )
        subagent = Subagent(
            config=config,
            tool_registry=self.tool_registry,
            hooks=self.hooks,
            parent_model=parent_model,
        )
        self._active[subagent.id] = subagent
        logger.info("Spawning subagent %s (%s) for task", subagent.id, config_name)
        result = subagent.execute(task_description, llm_callback=llm_callback)
        self._active.pop(subagent.id, None)
        self._completed.append(result)
        return result

    def get_active(self) -> list[dict[str, Any]]:
        return [s.stats() for s in self._active.values()]

    def get_completed(self) -> list[SubagentResult]:
        return list(self._completed)

    def cancel(self, subagent_id: str) -> bool:
        subagent = self._active.get(subagent_id)
        if subagent:
            subagent.cancel()
            return True
        return False

    def stats(self) -> dict[str, Any]:
        return {
            "registered_configs": len(self._configs),
            "active_count": len(self._active),
            "completed_count": len(self._completed),
            "successful_count": sum(1 for r in self._completed if r.success),
            "failed_count": sum(1 for r in self._completed if r.status == SubagentStatus.FAILED),
            "timeout_count": sum(1 for r in self._completed if r.status == SubagentStatus.TIMEOUT),
        }
