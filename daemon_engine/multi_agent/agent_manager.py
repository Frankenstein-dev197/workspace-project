"""Agent Manager: lifecycle management for a fleet of agents.

Inspired by DeerFlow's subagent registry and Ruflo's swarm management.
Handles agent registration, configuration, health monitoring, and pool
management.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from daemon_engine.core.agent_engine import Agent, AgentConfig
from daemon_engine.models.base import BaseLLM, get_default_llm
from daemon_engine.multi_agent.communication_system import CommunicationSystem

logger = logging.getLogger(__name__)


@dataclass
class AgentRecord:
    agent: Agent
    role: str = "worker"
    status: str = "idle"
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    tasks_completed: int = 0
    tasks_failed: int = 0


class AgentManager:
    """Manages a pool of agents with role-based assignment and health tracking."""

    def __init__(
        self,
        llm: BaseLLM | None = None,
        tool_registry: Any | None = None,
        communication: CommunicationSystem | None = None,
        memory: Any | None = None,
    ) -> None:
        self.llm = llm or get_default_llm()
        self.tool_registry = tool_registry
        self.communication = communication or CommunicationSystem()
        self.memory = memory
        self._agents: dict[str, AgentRecord] = {}
        self._role_templates: dict[str, AgentConfig] = self._default_roles()

    def _default_roles(self) -> dict[str, AgentConfig]:
        return {
            "researcher": AgentConfig(
                name="researcher",
                description="Finds and collects information from various sources",
                system_prompt="You are a research agent. Find relevant information, verify sources, and summarize findings.",
                tools=["web_search", "web_scraper", "osint_lookup"],
            ),
            "coder": AgentConfig(
                name="coder",
                description="Writes, reviews, and debugs code",
                system_prompt="You are a coding agent. Write clean, efficient code. Test your solutions.",
                tools=["bash", "file_write", "file_read"],
            ),
            "analyst": AgentConfig(
                name="analyst",
                description="Analyzes data and provides insights",
                system_prompt="You are an analysis agent. Examine data, identify patterns, and provide actionable insights.",
                tools=["bash", "data_query"],
            ),
            "devops": AgentConfig(
                name="devops",
                description="Handles deployment and infrastructure",
                system_prompt="You are a DevOps agent. Manage infrastructure, deployments, and automation.",
                tools=["bash", "ansible_playbook", "docker_build"],
            ),
            "orchestrator": AgentConfig(
                name="orchestrator",
                description="Coordinates other agents and manages workflows",
                system_prompt="You are an orchestrator agent. Break down complex tasks, assign subtasks to agents, and aggregate results.",
                max_turns=50,
            ),
        }

    def register_role(self, role_name: str, config: AgentConfig) -> None:
        self._role_templates[role_name] = config
        logger.info("Registered role template: %s", role_name)

    def spawn_agent(self, role: str = "worker", config: AgentConfig | None = None) -> Agent:
        if config is None:
            config = self._role_templates.get(role, AgentConfig(name=role))
        agent = Agent(
            config=config,
            llm=self.llm,
            tool_registry=self.tool_registry,
            memory=self.memory,
        )
        self.communication.register_agent(agent.id)
        record = AgentRecord(agent=agent, role=role)
        self._agents[agent.id] = record
        logger.info("Spawned %s agent: %s (%s)", role, agent.name, agent.id)
        return agent

    def get_agent(self, agent_id: str) -> Agent | None:
        record = self._agents.get(agent_id)
        return record.agent if record else None

    def get_agents_by_role(self, role: str) -> list[Agent]:
        return [r.agent for r in self._agents.values() if r.role == role]

    def get_available_agent(self, role: str | None = None) -> Agent | None:
        candidates = [
            r for r in self._agents.values()
            if r.status == "idle" and (role is None or r.role == role)
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda r: r.tasks_completed)
        return candidates[0].agent

    def update_status(self, agent_id: str, status: str) -> None:
        record = self._agents.get(agent_id)
        if record:
            record.status = status
            record.last_active = time.time()

    def record_completion(self, agent_id: str, success: bool) -> None:
        record = self._agents.get(agent_id)
        if record:
            if success:
                record.tasks_completed += 1
            else:
                record.tasks_failed += 1
            record.last_active = time.time()

    def terminate_agent(self, agent_id: str) -> bool:
        record = self._agents.pop(agent_id, None)
        if record:
            record.agent.cancel()
            self.communication.unregister_agent(agent_id)
            logger.info("Terminated agent %s", agent_id)
            return True
        return False

    def list_agents(self) -> list[AgentRecord]:
        return list(self._agents.values())

    def health_check(self) -> dict[str, Any]:
        total = len(self._agents)
        idle = sum(1 for r in self._agents.values() if r.status == "idle")
        active = sum(1 for r in self._agents.values() if r.status in ("running", "active"))
        return {
            "total_agents": total,
            "idle": idle,
            "active": active,
            "total_completed": sum(r.tasks_completed for r in self._agents.values()),
            "total_failed": sum(r.tasks_failed for r in self._agents.values()),
        }

    def terminate_all(self) -> None:
        for agent_id in list(self._agents.keys()):
            self.terminate_agent(agent_id)
