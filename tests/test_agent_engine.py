"""Tests for the core agent engine."""

import pytest

from daemon_engine.core.agent_engine import Agent, AgentConfig, AgentEngine, AgentState
from daemon_engine.core.task_planner import Task, TaskPriority
from daemon_engine.models.providers import MockProvider
from daemon_engine.tools.tool_registry import ToolRegistry


@pytest.fixture
def mock_llm():
    return MockProvider()


@pytest.fixture
def tool_registry():
    registry = ToolRegistry()
    registry.register("echo", "Echo tool", lambda **kw: f"Echo: {kw.get('query', '')}", category="test")
    return registry


@pytest.fixture
def agent(mock_llm, tool_registry):
    config = AgentConfig(
        name="test-agent",
        description="Test agent",
        max_turns=5,
        timeout_seconds=30,
        tools=["echo"],
    )
    return Agent(config=config, llm=mock_llm, tool_registry=tool_registry)


class TestAgentConfig:
    def test_default_config(self):
        config = AgentConfig()
        assert config.name == "default-agent"
        assert config.max_turns == 25
        assert config.timeout_seconds == 300

    def test_custom_config(self):
        config = AgentConfig(name="custom", max_turns=10)
        assert config.name == "custom"
        assert config.max_turns == 10


class TestAgentState:
    def test_is_terminal(self):
        assert AgentState.COMPLETED.is_terminal
        assert AgentState.FAILED.is_terminal
        assert AgentState.CANCELLED.is_terminal
        assert not AgentState.IDLE.is_terminal
        assert not AgentState.THINKING.is_terminal


class TestAgent:
    def test_agent_creation(self, agent):
        assert agent.name == "test-agent"
        assert agent.state == AgentState.IDLE
        assert agent.id  # UUID generated

    def test_agent_reset(self, agent):
        agent.state = AgentState.THINKING
        agent.history.append(type(agent.history[0])(turn=1) if agent.history else None)
        agent.reset()
        assert agent.state == AgentState.IDLE
        assert len(agent.history) == 0

    def test_agent_cancel(self, agent):
        agent.cancel()
        assert agent.state == AgentState.CANCELLED

    def test_agent_run_task(self, agent):
        task = Task(description="Test task", priority=TaskPriority.MEDIUM)
        result = agent.run(task)
        assert result.agent_id == agent.id
        assert result.status in (AgentState.COMPLETED, AgentState.FAILED)
        assert result.task_id == task.id

    def test_agent_run_completes(self, agent):
        task = Task(description="Complete this task and give a final answer")
        result = agent.run(task)
        assert result.num_turns > 0


class TestAgentEngine:
    def test_create_agent(self, mock_llm, tool_registry):
        engine = AgentEngine(llm=mock_llm, tool_registry=tool_registry)
        agent = engine.create_agent(config=AgentConfig(name="engine-test"))
        assert agent.name == "engine-test"
        assert agent.id in [a.id for a in engine.list_agents()]

    def test_get_agent(self, mock_llm, tool_registry):
        engine = AgentEngine(llm=mock_llm, tool_registry=tool_registry)
        agent = engine.create_agent()
        fetched = engine.get_agent(agent.id)
        assert fetched is agent

    def test_remove_agent(self, mock_llm, tool_registry):
        engine = AgentEngine(llm=mock_llm, tool_registry=tool_registry)
        agent = engine.create_agent()
        assert engine.remove_agent(agent.id) is True
        assert engine.get_agent(agent.id) is None

    def test_run_task(self, mock_llm, tool_registry):
        engine = AgentEngine(llm=mock_llm, tool_registry=tool_registry)
        task = Task(description="Simple test task")
        result = engine.run_task(task)
        assert result.status in (AgentState.COMPLETED, AgentState.FAILED)
