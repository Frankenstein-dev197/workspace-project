"""Daemon Engine: the main entry point that connects all subsystems.

This module wires together the core engine, multi-agent system, memory,
tools, runtime, and model integration into a single functional AI engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from daemon_engine.core.agent_engine import Agent, AgentConfig, AgentEngine, AgentResult
from daemon_engine.core.decision_system import DecisionSystem, DecisionStrategy
from daemon_engine.core.reasoning_engine import ReasoningEngine, ReasoningStrategy
from daemon_engine.core.task_planner import Task, TaskPlanner, TaskPriority, TaskStatus
from daemon_engine.memory.unified import UnifiedMemory
from daemon_engine.models.base import BaseLLM, LLMConfig, get_default_llm
from daemon_engine.multi_agent.agent_manager import AgentManager
from daemon_engine.multi_agent.communication_system import CommunicationSystem
from daemon_engine.multi_agent.orchestrator import Orchestrator, Workflow
from daemon_engine.runtime.sandbox import Sandbox, SandboxConfig
from daemon_engine.runtime.virtual_computer_engine import VirtualComputerEngine
from daemon_engine.tools.automation_tools import AutomationTools
from daemon_engine.tools.browser_tools import BrowserTools
from daemon_engine.tools.devops_tools import DevOpsTools
from daemon_engine.tools.research_tools import ResearchTools
from daemon_engine.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class DaemonEngine:
    """The unified agentic AI engine.

    A single object that ties together agents, reasoning, memory, tools,
    runtime, and multi-agent orchestration. This is the top-level interface
    for building and running agentic workflows.
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        memory_path: str | Path | None = None,
        workdir: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or {}
        self.llm = llm or get_default_llm()
        self.memory = UnifiedMemory(storage_path=memory_path)
        self.tool_registry = ToolRegistry()
        self._register_default_tools(workdir)
        self.communication = CommunicationSystem()
        self.agent_manager = AgentManager(
            llm=self.llm,
            tool_registry=self.tool_registry,
            communication=self.communication,
            memory=self.memory,
        )
        self.agent_engine = AgentEngine(
            llm=self.llm, tool_registry=self.tool_registry, memory=self.memory
        )
        self.task_planner = TaskPlanner(llm=self.llm)
        self.reasoning_engine = ReasoningEngine(llm=self.llm)
        self.decision_system = DecisionSystem(llm=self.llm)
        sandbox_config = SandboxConfig(workdir=workdir) if workdir else SandboxConfig()
        self.sandbox = Sandbox(config=sandbox_config)
        self.virtual_computer = VirtualComputerEngine(config=sandbox_config)
        self.orchestrator = Orchestrator(
            llm=self.llm,
            agent_manager=self.agent_manager,
            task_planner=self.task_planner,
            communication=self.communication,
            decision_system=self.decision_system,
            tool_registry=self.tool_registry,
            memory=self.memory,
        )
        logger.info("DaemonEngine initialized — LLM=%s, tools=%d", self.llm.model_name, len(self.tool_registry.list_tools()))

    def _register_default_tools(self, workdir: str | Path | None = None) -> None:
        BrowserTools().register_all(self.tool_registry)
        ResearchTools().register_all(self.tool_registry)
        DevOpsTools(workdir=workdir).register_all(self.tool_registry)
        AutomationTools().register_all(self.tool_registry)

    def run_task(self, description: str, agent_config: AgentConfig | None = None) -> AgentResult:
        task = Task(description=description, priority=TaskPriority.HIGH)
        return self.agent_engine.run_task(task, agent_config=agent_config)

    def run_goal(self, goal: str, max_tasks: int = 20) -> Workflow:
        return self.orchestrator.execute_goal(goal, max_tasks=max_tasks)

    def reason(self, problem: str, strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT) -> Any:
        return self.reasoning_engine.reason(problem, strategy=strategy)

    def decide(self, options: list, context: dict | None = None, strategy: DecisionStrategy | None = None) -> Any:
        return self.decision_system.decide(options, context=context or {}, strategy=strategy)

    def plan_tasks(self, goal: str, max_depth: int = 3) -> str:
        self.task_planner.decompose(goal, max_depth=max_depth)
        return self.task_planner.get_task_tree()

    def create_agent(self, role: str = "worker", config: AgentConfig | None = None) -> Agent:
        if config is None:
            return self.agent_manager.spawn_agent(role=role)
        return self.agent_engine.create_agent(config=config)

    def execute_code(self, code: str, timeout: int | None = None) -> Any:
        return self.virtual_computer.execute_code(code, timeout=timeout)

    def execute_command(self, command: str, timeout: int | None = None) -> Any:
        return self.virtual_computer.execute(command, timeout=timeout)

    def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        return self.tool_registry.execute(tool_name, **kwargs)

    def remember(self, content: str, memory_type: str = "episodic", **kwargs: Any) -> str:
        return self.memory.store(content=content, memory_type=memory_type, **kwargs)

    def recall(self, query: str, limit: int = 5) -> str:
        return self.memory.recall(query, limit=limit)

    def save_state(self) -> None:
        self.memory.save_all()

    def system_status(self) -> dict[str, Any]:
        return {
            "llm_model": self.llm.model_name,
            "tools_available": self.tool_registry.list_tools(),
            "tool_count": len(self.tool_registry.list_tools()),
            "agents": self.agent_manager.health_check(),
            "memory": self.memory.stats(),
            "virtual_computer": self.virtual_computer.system_info(),
            "workflows": len(self.orchestrator.list_workflows()),
            "tasks": len(self.task_planner.all_tasks()),
        }

    def shutdown(self) -> None:
        self.save_state()
        self.agent_manager.terminate_all()
        self.virtual_computer.shutdown()
        logger.info("DaemonEngine shut down cleanly")
