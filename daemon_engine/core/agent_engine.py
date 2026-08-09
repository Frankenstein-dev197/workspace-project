"""Agent Engine: autonomous agent execution loop.

Integrates the agent-loop pattern from learn-claude-code with the subagent
execution model from DeerFlow and the autonomous goal-seeking loop from AutoGPT.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from daemon_engine.core.task_planner import Task, TaskStatus
from daemon_engine.models.base import BaseLLM, get_default_llm

logger = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {AgentState.COMPLETED, AgentState.FAILED, AgentState.CANCELLED}


@dataclass
class AgentConfig:
    name: str = "default-agent"
    description: str = "A general-purpose autonomous agent"
    system_prompt: str = (
        "You are an autonomous AI agent. Break down complex tasks into steps, "
        "use available tools, and work toward the goal until completion."
    )
    model: str | None = None
    max_turns: int = 25
    timeout_seconds: int = 300
    tools: list[str] = field(default_factory=list)
    memory_enabled: bool = True
    reasoning_enabled: bool = True


@dataclass
class AgentStep:
    turn: int
    thought: str = ""
    action: str = ""
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    agent_id: str
    task_id: str
    status: AgentState
    output: str = ""
    steps: list[AgentStep] = field(default_factory=list)
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None

    @property
    def duration(self) -> float:
        end = self.completed_at or time.time()
        return end - self.started_at

    @property
    def num_turns(self) -> int:
        return len(self.steps)


class Agent:
    """A single autonomous agent with a think-act-observe loop."""

    def __init__(
        self,
        config: AgentConfig | None = None,
        llm: BaseLLM | None = None,
        tool_registry: Any | None = None,
        memory: Any | None = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.llm = llm or get_default_llm()
        self.tool_registry = tool_registry
        self.memory = memory
        self.id: str = str(uuid.uuid4())
        self.state: AgentState = AgentState.IDLE
        self.history: list[AgentStep] = []
        self._messages: list[dict[str, str]] = []

    @property
    def name(self) -> str:
        return self.config.name

    def _build_system_prompt(self, task: Task) -> str:
        prompt = self.config.system_prompt
        if self.config.tools and self.tool_registry:
            available = self.tool_registry.list_tools()
            tool_names = [t for t in available if t in self.config.tools] if self.config.tools else available
            prompt += f"\n\nAvailable tools: {', '.join(tool_names)}"
        prompt += f"\n\nCurrent task: {task.description}"
        if self.memory and self.config.memory_enabled:
            context = self.memory.recall(task.description, limit=5)
            if context:
                prompt += f"\n\nRelevant memory:\n{context}"
        return prompt

    def _think(self, task: Task, observation: str | None = None) -> AgentStep:
        step = AgentStep(turn=len(self.history) + 1)
        self.state = AgentState.THINKING

        if not self._messages:
            self._messages.append({"role": "system", "content": self._build_system_prompt(task)})
            self._messages.append({"role": "user", "content": task.description})

        if observation:
            self._messages.append({"role": "user", "content": f"Observation: {observation}"})

        try:
            response = self.llm.chat(self._messages)
            step.thought = response
            self._messages.append({"role": "assistant", "content": response})
        except Exception as exc:
            logger.error("Agent %s LLM error: %s", self.name, exc)
            step.thought = f"Error during thinking: {exc}"
        return step

    def _act(self, step: AgentStep) -> str:
        self.state = AgentState.ACTING
        if not self.tool_registry:
            return "No tools available"
        tool_name, tool_input = self._parse_action(step.thought)
        if not tool_name:
            return "No action parsed — treating as final answer"
        step.action = tool_name
        step.action_input = tool_input
        try:
            result = self.tool_registry.execute(tool_name, **tool_input)
            step.observation = str(result)
            return step.observation
        except Exception as exc:
            logger.error("Tool execution error: %s", exc)
            step.observation = f"Tool error: {exc}"
            return step.observation

    def _parse_action(self, thought: str) -> tuple[str | None, dict[str, Any]]:
        thought_lower = thought.lower()
        for tool_name in (self.config.tools or []):
            if tool_name in thought_lower:
                return tool_name, {"query": thought}
        if "final answer:" in thought_lower or "task complete" in thought_lower:
            return None, {}
        if self.tool_registry:
            available = self.tool_registry.list_tools()
            for tool_name in available:
                if tool_name in thought_lower:
                    return tool_name, {"query": thought}
        return None, {}

    def _observe(self, step: AgentStep, observation: str) -> None:
        self.state = AgentState.OBSERVING
        step.observation = observation
        self.history.append(step)
        if self.memory and self.config.memory_enabled:
            self.memory.store(
                content=f"Step {step.turn}: {step.thought[:200]} -> {step.observation[:200]}",
                metadata={"agent_id": self.id, "turn": step.turn},
            )

    def _is_complete(self, thought: str) -> bool:
        markers = ["final answer:", "task complete", "done.", "finished", "goal achieved"]
        return any(m in thought.lower() for m in markers)

    def run(self, task: Task) -> AgentResult:
        result = AgentResult(agent_id=self.id, task_id=task.id, status=AgentState.IDLE)
        task.status = TaskStatus.IN_PROGRESS
        start = time.time()
        observation: str | None = None

        try:
            for turn in range(self.config.max_turns):
                if time.time() - start > self.config.timeout_seconds:
                    result.status = AgentState.FAILED
                    result.error = "Timeout exceeded"
                    break
                step = self._think(task, observation)
                if self._is_complete(step.thought):
                    self._observe(step, "Task marked complete by agent")
                    result.output = step.thought
                    result.status = AgentState.COMPLETED
                    break
                observation = self._act(step)
                self._observe(step, observation)
            else:
                result.status = AgentState.FAILED
                result.error = f"Max turns ({self.config.max_turns}) exceeded"
        except Exception as exc:
            logger.exception("Agent %s failed", self.name)
            result.status = AgentState.FAILED
            result.error = str(exc)
        finally:
            result.steps = self.history
            result.completed_at = time.time()
            if result.status == AgentState.COMPLETED:
                task.status = TaskStatus.COMPLETED
            elif result.status == AgentState.FAILED:
                task.status = TaskStatus.FAILED
        return result

    def cancel(self) -> None:
        self.state = AgentState.CANCELLED

    def reset(self) -> None:
        self.state = AgentState.IDLE
        self.history.clear()
        self._messages.clear()


class AgentEngine:
    """Manages creation, execution, and lifecycle of agents.

    Inspired by DeerFlow's SubagentExecutor — supports concurrent agent
    execution, registration, and result aggregation.
    """

    def __init__(self, llm: BaseLLM | None = None, tool_registry: Any | None = None, memory: Any | None = None) -> None:
        self.llm = llm or get_default_llm()
        self.tool_registry = tool_registry
        self.memory = memory
        self._agents: dict[str, Agent] = {}

    def create_agent(self, config: AgentConfig | None = None) -> Agent:
        agent = Agent(config=config, llm=self.llm, tool_registry=self.tool_registry, memory=self.memory)
        self._agents[agent.id] = agent
        logger.info("Created agent %s (%s)", agent.name, agent.id)
        return agent

    def get_agent(self, agent_id: str) -> Agent | None:
        return self._agents.get(agent_id)

    def list_agents(self) -> list[Agent]:
        return list(self._agents.values())

    def run_task(self, task: Task, agent_config: AgentConfig | None = None) -> AgentResult:
        agent = self.create_agent(config=agent_config)
        return agent.run(task)

    def remove_agent(self, agent_id: str) -> bool:
        return self._agents.pop(agent_id, None) is not None
