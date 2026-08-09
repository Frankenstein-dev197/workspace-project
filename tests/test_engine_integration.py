"""Integration tests for the full DaemonEngine."""

import pytest

from daemon_engine import DaemonEngine
from daemon_engine.core.agent_engine import AgentConfig
from daemon_engine.core.reasoning_engine import ReasoningStrategy
from daemon_engine.core.task_planner import TaskPriority


@pytest.fixture
def engine():
    eng = DaemonEngine()
    yield eng
    eng.shutdown()


class TestDaemonEngineInit:
    def test_engine_creates(self, engine):
        assert engine.llm is not None
        assert engine.memory is not None
        assert engine.tool_registry is not None
        assert engine.agent_manager is not None
        assert engine.orchestrator is not None
        assert engine.reasoning_engine is not None
        assert engine.decision_system is not None
        assert engine.sandbox is not None
        assert engine.virtual_computer is not None

    def test_tools_registered(self, engine):
        tools = engine.tool_registry.list_tools()
        assert len(tools) >= 15  # browser + research + devops + automation tools
        assert "web_search" in tools
        assert "bash" in tools
        assert "web_scraper" in tools
        assert "run_pipeline" in tools

    def test_system_status(self, engine):
        status = engine.system_status()
        assert "llm_model" in status
        assert "tools_available" in status
        assert "agents" in status
        assert "memory" in status
        assert "virtual_computer" in status


class TestDaemonEngineTask:
    def test_run_task(self, engine):
        result = engine.run_task("Say hello and give a final answer")
        assert result.status.value in ("completed", "failed")
        assert result.num_turns > 0

    def test_create_agent(self, engine):
        agent = engine.create_agent(role="researcher")
        assert agent.name == "researcher"
        assert agent.id is not None


class TestDaemonEngineReasoning:
    def test_reason(self, engine):
        result = engine.reason("How to solve this problem?", ReasoningStrategy.CHAIN_OF_THOUGHT)
        assert result.strategy.value == "chain_of_thought"
        assert len(result.steps) > 0


class TestDaemonEnginePlanning:
    def test_plan_tasks(self, engine):
        tree = engine.plan_tasks("Build a REST API", max_depth=2)
        assert "Build a REST API" in tree
        assert len(engine.task_planner.all_tasks()) > 0


class TestDaemonEngineTools:
    def test_use_tool(self, engine):
        result = engine.use_tool("web_search", query="python")
        assert result.success is True

    def test_use_tool_bash(self, engine):
        result = engine.use_tool("bash", command="echo 'integration test'")
        assert result.success is True
        assert "integration test" in result.output


class TestDaemonEngineRuntime:
    def test_execute_code(self, engine):
        result = engine.execute_code("print('engine code execution')")
        assert result.result is not None
        assert result.result.success is True

    def test_execute_command(self, engine):
        result = engine.execute_command("echo 'engine command'")
        assert result.result is not None
        assert result.result.success is True


class TestDaemonEngineMemory:
    def test_remember_and_recall(self, engine):
        engine.remember("The project uses Python 3.12", memory_type="semantic")
        result = engine.recall("Python")
        assert result  # Should find something

    def test_recall_empty(self, engine):
        result = engine.recall("nonexistent_topic_xyz123")
        # May be empty or have some mock results
        assert isinstance(result, str)


class TestDaemonEngineWorkflow:
    def test_run_goal(self, engine):
        workflow = engine.run_goal("Research Python web frameworks", max_tasks=3)
        assert workflow.goal == "Research Python web frameworks"
        assert workflow.is_complete
        assert len(workflow.tasks) > 0

    def test_workflow_summary(self, engine):
        workflow = engine.run_goal("Write a simple script", max_tasks=2)
        summary = engine.orchestrator.get_workflow_summary(workflow.id)
        assert summary["goal"] == "Write a simple script"
        assert "status" in summary
