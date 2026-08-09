"""Tests for the multi-agent system."""

import pytest

from daemon_engine.core.agent_engine import AgentConfig
from daemon_engine.models.providers import MockProvider
from daemon_engine.multi_agent.agent_manager import AgentManager
from daemon_engine.multi_agent.communication_system import CommunicationSystem, Message, MessageType
from daemon_engine.multi_agent.orchestrator import Orchestrator
from daemon_engine.tools.tool_registry import ToolRegistry


@pytest.fixture
def communication():
    return CommunicationSystem()


@pytest.fixture
def agent_manager(communication):
    return AgentManager(
        llm=MockProvider(),
        tool_registry=ToolRegistry(),
        communication=communication,
    )


class TestCommunicationSystem:
    def test_register_agent(self, communication):
        communication.register_agent("agent-1")
        msg = Message(sender_id="agent-1", recipient_id="agent-1", content="self test")
        assert communication.send(msg) is True

    def test_send_direct(self, communication):
        communication.register_agent("agent-1")
        communication.register_agent("agent-2")
        msg = Message(sender_id="agent-1", recipient_id="agent-2", content="hello")
        assert communication.send(msg) is True
        received = communication.receive("agent-2")
        assert received is not None
        assert received.content == "hello"

    def test_send_broadcast(self, communication):
        communication.register_agent("agent-1")
        communication.register_agent("agent-2")
        communication.register_agent("agent-3")
        msg = Message(sender_id="agent-1", msg_type=MessageType.BROADCAST, content="broadcast msg")
        communication.send(msg)
        assert communication.receive("agent-2") is not None
        assert communication.receive("agent-3") is not None
        assert communication.receive("agent-1") is None  # sender doesn't receive

    def test_send_to_unknown(self, communication):
        msg = Message(sender_id="agent-1", recipient_id="unknown", content="test")
        assert communication.send(msg) is False

    def test_shared_state(self, communication):
        communication.set_shared_state("key", "value")
        assert communication.get_shared_state("key") == "value"

    def test_message_log(self, communication):
        communication.register_agent("a")
        communication.register_agent("b")
        communication.send(Message(sender_id="a", recipient_id="b", content="msg1"))
        log = communication.get_message_log()
        assert len(log) == 1

    def test_subscribe(self, communication):
        communication.register_agent("agent-1")
        received: list[Message] = []
        communication.subscribe("agent-1", lambda m: received.append(m))
        communication.send(Message(sender_id="sender", recipient_id="agent-1", content="test"))
        assert len(received) == 1


class TestAgentManager:
    def test_spawn_agent(self, agent_manager):
        agent = agent_manager.spawn_agent(role="researcher")
        assert agent.name == "researcher"
        assert agent.id in [r.agent.id for r in agent_manager.list_agents()]

    def test_get_agents_by_role(self, agent_manager):
        agent_manager.spawn_agent(role="researcher")
        agent_manager.spawn_agent(role="coder")
        researchers = agent_manager.get_agents_by_role("researcher")
        assert len(researchers) == 1

    def test_get_available_agent(self, agent_manager):
        agent = agent_manager.spawn_agent(role="worker")
        available = agent_manager.get_available_agent(role="worker")
        assert available is agent

    def test_terminate_agent(self, agent_manager):
        agent = agent_manager.spawn_agent()
        assert agent_manager.terminate_agent(agent.id) is True
        assert agent_manager.get_agent(agent.id) is None

    def test_health_check(self, agent_manager):
        agent_manager.spawn_agent(role="worker")
        health = agent_manager.health_check()
        assert health["total_agents"] == 1
        assert health["idle"] == 1

    def test_default_roles(self, agent_manager):
        assert "researcher" in agent_manager._role_templates
        assert "coder" in agent_manager._role_templates
        assert "devops" in agent_manager._role_templates

    def test_register_custom_role(self, agent_manager):
        config = AgentConfig(name="custom_role", description="Custom")
        agent_manager.register_role("custom", config)
        agent = agent_manager.spawn_agent(role="custom")
        assert agent.name == "custom_role"


class TestOrchestrator:
    @pytest.fixture
    def orchestrator(self, agent_manager):
        return Orchestrator(
            llm=MockProvider(),
            agent_manager=agent_manager,
            task_planner=agent_manager and None,  # will use default
        )

    def test_execute_goal(self):
        orch = Orchestrator(llm=MockProvider(), tool_registry=ToolRegistry())
        try:
            workflow = orch.execute_goal("Build a simple web scraper", max_tasks=5)
            assert workflow.goal == "Build a simple web scraper"
            assert workflow.is_complete
        finally:
            orch.agent_manager.terminate_all()

    def test_get_workflow_summary(self):
        orch = Orchestrator(llm=MockProvider(), tool_registry=ToolRegistry())
        try:
            workflow = orch.execute_goal("Write a test", max_tasks=3)
            summary = orch.get_workflow_summary(workflow.id)
            assert summary["goal"] == "Write a test"
            assert "status" in summary
            assert "progress" in summary
        finally:
            orch.agent_manager.terminate_all()

    def test_list_workflows(self):
        orch = Orchestrator(llm=MockProvider(), tool_registry=ToolRegistry())
        try:
            orch.execute_goal("Task 1", max_tasks=2)
            orch.execute_goal("Task 2", max_tasks=2)
            assert len(orch.list_workflows()) == 2
        finally:
            orch.agent_manager.terminate_all()
