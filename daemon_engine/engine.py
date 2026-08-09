"""Daemon Engine: the main entry point that connects all subsystems.

This module wires together the core engine, multi-agent system, memory,
skills, knowledge, tools, runtime, models, and infrastructure into a
single functional AI engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from daemon_engine.core.agent_engine import Agent, AgentConfig, AgentEngine, AgentResult
from daemon_engine.core.decision_system import DecisionSystem, DecisionStrategy
from daemon_engine.core.hooks import HookRegistry, HookEvent, create_default_registry
from daemon_engine.core.message_manager import MessageManager, MessageRole
from daemon_engine.core.reasoning_engine import ReasoningEngine, ReasoningStrategy
from daemon_engine.core.security import SecurityManager
from daemon_engine.core.task_planner import Task, TaskPlanner, TaskPriority, TaskStatus
from daemon_engine.infrastructure.deployment_manager import DeploymentManager, DeployConfig, DeployTarget
from daemon_engine.infrastructure.docker_manager import DockerManager
from daemon_engine.knowledge.algorithm_patterns import AlgorithmPatternLibrary
from daemon_engine.knowledge.devops_knowledge import DevOpsKnowledgeBase
from daemon_engine.knowledge.knowledge_base import KnowledgeBase
from daemon_engine.memory.unified import UnifiedMemory
from daemon_engine.models.base import BaseLLM, LLMConfig, get_default_llm
from daemon_engine.multi_agent.agent_manager import AgentManager
from daemon_engine.multi_agent.communication_system import CommunicationSystem
from daemon_engine.multi_agent.orchestrator import Orchestrator, Workflow
from daemon_engine.multi_agent.swarm import Swarm, SwarmManager, SwarmConfig, SwarmTopology
from daemon_engine.runtime.code_execution.executor import CodeExecutor
from daemon_engine.runtime.firecracker import FirecrackerManager
from daemon_engine.runtime.sandbox import Sandbox, SandboxConfig
from daemon_engine.runtime.virtual_computer_engine import VirtualComputerEngine
from daemon_engine.skills.skill_registry import SkillRegistry
from daemon_engine.tools.automation_tools import AutomationTools
from daemon_engine.tools.browser_tools import BrowserTools
from daemon_engine.tools.devops_tools import DevOpsTools
from daemon_engine.tools.research_tools import ResearchTools
from daemon_engine.tools.scraping_tools import ScrapingTools
from daemon_engine.tools.tool_registry import ToolRegistry

logger = logging.getLogger(__name__)


class DaemonEngine:
    """The unified agentic AI engine.

    A single object that ties together agents, reasoning, memory, skills,
    knowledge, tools, runtime, infrastructure, and multi-agent orchestration.
    This is the top-level interface for building and running agentic workflows.
    """

    def __init__(
        self,
        llm: BaseLLM | None = None,
        memory_path: str | Path | None = None,
        workdir: str | Path | None = None,
        config: dict[str, Any] | None = None,
        enable_firecracker: bool = False,
        enable_docker: bool = True,
        skills_paths: list[str | Path] | None = None,
    ) -> None:
        self.config = config or {}
        self.llm = llm or get_default_llm()
        self.memory = UnifiedMemory(storage_path=memory_path)
        self.tool_registry = ToolRegistry()
        self._register_default_tools(workdir)
        self.hooks = create_default_registry()
        self.security = SecurityManager()
        self.swarm_manager = SwarmManager()
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
        self.knowledge_base = KnowledgeBase(storage_path=memory_path)
        self.algorithm_patterns = AlgorithmPatternLibrary(knowledge_base=self.knowledge_base)
        self.algorithm_patterns.load_patterns()
        self.devops_knowledge = DevOpsKnowledgeBase(knowledge_base=self.knowledge_base)
        self.devops_knowledge.load_knowledge()
        self.skill_registry = SkillRegistry()
        if skills_paths:
            for path in skills_paths:
                self.skill_registry.add_path(path)
        else:
            self.skill_registry.discover()
        sandbox_config = SandboxConfig(workdir=workdir) if workdir else SandboxConfig()
        self.sandbox = Sandbox(config=sandbox_config)
        self.virtual_computer = VirtualComputerEngine(config=sandbox_config)
        self.code_executor = CodeExecutor(base_workdir=workdir)
        self.firecracker = FirecrackerManager() if enable_firecracker else None
        self.docker = DockerManager() if enable_docker else None
        self.deployment_manager = DeploymentManager()
        self.orchestrator = Orchestrator(
            llm=self.llm,
            agent_manager=self.agent_manager,
            task_planner=self.task_planner,
            communication=self.communication,
            decision_system=self.decision_system,
            tool_registry=self.tool_registry,
            memory=self.memory,
        )
        logger.info(
            "DaemonEngine initialized — LLM=%s, tools=%d, skills=%d, knowledge=%d",
            self.llm.model_name,
            len(self.tool_registry.list_tools()),
            self.skill_registry.stats()["total_skills"],
            self.knowledge_base.stats()["total_entries"],
        )

    def _register_default_tools(self, workdir: str | Path | None = None) -> None:
        BrowserTools().register_all(self.tool_registry)
        ResearchTools().register_all(self.tool_registry)
        ScrapingTools().register_all(self.tool_registry)
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

    def execute_code(self, code: str, language: str = "python", timeout: int = 30) -> Any:
        return self.code_executor.execute(code, language=language, timeout=timeout)

    def execute_command(self, command: str, timeout: int | None = None) -> Any:
        return self.virtual_computer.execute(command, timeout=timeout)

    def use_tool(self, tool_name: str, **kwargs: Any) -> Any:
        return self.tool_registry.execute(tool_name, **kwargs)

    def remember(self, content: str, memory_type: str = "episodic", **kwargs: Any) -> str:
        return self.memory.store(content=content, memory_type=memory_type, **kwargs)

    def recall(self, query: str, limit: int = 5) -> str:
        return self.memory.recall(query, limit=limit)

    def search_knowledge(self, query: str, limit: int = 5) -> list[str]:
        results = self.knowledge_base.search(query, limit=limit)
        return [f"{entry.title}: {entry.content[:200]}" for _, entry in results]

    def activate_skill(self, name: str) -> bool:
        return self.skill_registry.activate(name)

    def get_skill_context(self, names: list[str] | None = None) -> str:
        return self.skill_registry.get_context(names)

    def search_skills(self, query: str) -> list[str]:
        skills = self.skill_registry.search(query)
        return [f"{s.name}: {s.description[:100]}" for s in skills]

    def create_vm(self, vcpus: int = 2, mem_mib: int = 512) -> Any:
        if not self.firecracker:
            raise RuntimeError("Firecracker not enabled. Set enable_firecracker=True.")
        vm = self.firecracker.create_vm(vcpus=vcpus, mem_mib=mem_mib)
        vm.start()
        return vm

    def deploy(self, project_path: str, target: str = "local", **kwargs: Any) -> Any:
        deploy_target = DeployTarget(target) if isinstance(target, str) else target
        config = DeployConfig(project_path=project_path, target=deploy_target, **kwargs)
        return self.deployment_manager.deploy(config)

    def create_swarm(
        self,
        topology: str = "hierarchical-mesh",
        max_agents: int = 15,
        consensus: str = "majority",
    ) -> Swarm:
        topo = SwarmTopology(topology) if isinstance(topology, str) else topology
        from daemon_engine.multi_agent.swarm import ConsensusMechanism
        config = SwarmConfig(
            topology=topo,
            max_agents=max_agents,
            consensus_mechanism=ConsensusMechanism(consensus),
        )
        return self.swarm_manager.create_swarm(config=config)

    def register_hook(self, event: str, callback: Any) -> None:
        hook_event = HookEvent(event) if isinstance(event, str) else event
        self.hooks.register(hook_event, callback)

    def list_hooks(self) -> dict[str, list[str]]:
        return self.hooks.list_hooks()

    def save_state(self) -> None:
        self.memory.save_all()
        self.knowledge_base.save()

    def system_status(self) -> dict[str, Any]:
        status: dict[str, Any] = {
            "llm_model": self.llm.model_name,
            "tools_available": self.tool_registry.list_tools(),
            "tool_count": len(self.tool_registry.list_tools()),
            "agents": self.agent_manager.health_check(),
            "memory": self.memory.stats(),
            "virtual_computer": self.virtual_computer.system_info(),
            "workflows": len(self.orchestrator.list_workflows()),
            "tasks": len(self.task_planner.all_tasks()),
            "knowledge": self.knowledge_base.stats(),
            "skills": self.skill_registry.stats(),
            "code_executor": self.code_executor.info(),
            "deployments": self.deployment_manager.stats(),
            "hooks": self.hooks.stats(),
            "swarms": self.swarm_manager.stats(),
            "security": self.security.stats(),
        }
        if self.firecracker:
            status["firecracker"] = self.firecracker.stats()
        if self.docker:
            status["docker"] = self.docker.stats()
        return status

    def shutdown(self) -> None:
        self.save_state()
        self.agent_manager.terminate_all()
        self.virtual_computer.shutdown()
        self.code_executor.cleanup()
        if self.firecracker:
            self.firecracker.shutdown_all()
        logger.info("DaemonEngine shut down cleanly")
