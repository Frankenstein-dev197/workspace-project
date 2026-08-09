"""Tests for swarm coordination system."""

import pytest

from daemon_engine.multi_agent.swarm import (
    Swarm,
    SwarmManager,
    SwarmConfig,
    SwarmTopology,
    SwarmAgent,
    SwarmAgentRole,
    SwarmAgentStatus,
    ConsensusMechanism,
    FailureHandling,
)


class TestSwarmConfig:
    def test_defaults(self):
        config = SwarmConfig()
        assert config.topology == SwarmTopology.HIERARCHICAL_MESH
        assert config.max_agents == 15
        assert config.consensus_mechanism == ConsensusMechanism.MAJORITY

    def test_custom(self):
        config = SwarmConfig(
            topology=SwarmTopology.MESH,
            max_agents=10,
            consensus_mechanism=ConsensusMechanism.WEIGHTED,
        )
        assert config.topology == SwarmTopology.MESH
        assert config.max_agents == 10


class TestSwarmAgent:
    def test_is_available(self):
        agent = SwarmAgent(id="a1", status=SwarmAgentStatus.IDLE)
        assert agent.is_available() is True
        agent.status = SwarmAgentStatus.BUSY
        assert agent.is_available() is False

    def test_to_dict(self):
        agent = SwarmAgent(id="a1", type="coder", capabilities=["python"])
        d = agent.to_dict()
        assert d["id"] == "a1"
        assert d["type"] == "coder"
        assert "python" in d["capabilities"]


class TestSwarm:
    def test_add_agent(self):
        swarm = Swarm()
        agent = swarm.add_agent(agent_id="a1", agent_type="coder")
        assert agent.id == "a1"
        assert swarm.agent_count == 1

    def test_add_coordinator(self):
        swarm = Swarm()
        coord = swarm.add_agent(agent_id="c1", role=SwarmAgentRole.COORDINATOR)
        assert coord.role == SwarmAgentRole.COORDINATOR

    def test_remove_agent(self):
        swarm = Swarm()
        swarm.add_agent(agent_id="a1")
        assert swarm.remove_agent("a1") is True
        assert swarm.agent_count == 0

    def test_get_available_agent(self):
        swarm = Swarm()
        swarm.add_agent(agent_id="a1", capabilities=["python"])
        agent = swarm.get_available_agent(required_capability="python")
        assert agent is not None
        assert agent.id == "a1"

    def test_get_available_agent_none(self):
        swarm = Swarm()
        swarm.add_agent(agent_id="a1", capabilities=["python"])
        agent = swarm.get_available_agent(required_capability="rust")
        assert agent is None

    def test_assign_task(self):
        swarm = Swarm()
        swarm.add_agent(agent_id="a1")
        assert swarm.assign_task("a1", "task-1") is True
        agent = swarm.get_agent("a1")
        assert agent.status == SwarmAgentStatus.BUSY

    def test_complete_task(self):
        swarm = Swarm()
        swarm.add_agent(agent_id="a1")
        swarm.assign_task("a1", "task-1")
        swarm.complete_task("a1", "task-1", success=True)
        agent = swarm.get_agent("a1")
        assert agent.tasks_completed == 1
        assert agent.status == SwarmAgentStatus.IDLE

    def test_scale_up(self):
        swarm = Swarm(config=SwarmConfig(max_agents=20))
        swarm.add_agent(agent_id="a1")
        final = swarm.scale(5)
        assert final == 5

    def test_scale_down(self):
        swarm = Swarm()
        for i in range(5):
            swarm.add_agent(agent_id=f"a{i}")
        final = swarm.scale(2)
        assert final == 2

    def test_mesh_topology(self):
        swarm = Swarm(config=SwarmConfig(topology=SwarmTopology.MESH))
        swarm.add_agent(agent_id="a1")
        swarm.add_agent(agent_id="a2")
        swarm.add_agent(agent_id="a3")
        a1 = swarm.get_agent("a1")
        assert "a2" in a1.connections
        assert "a3" in a1.connections

    def test_hierarchical_topology(self):
        swarm = Swarm(config=SwarmConfig(topology=SwarmTopology.HIERARCHICAL))
        swarm.add_agent(agent_id="c1", role=SwarmAgentRole.COORDINATOR)
        swarm.add_agent(agent_id="w1")
        swarm.add_agent(agent_id="w2")
        coord = swarm.get_agent("c1")
        assert "w1" in coord.connections
        assert "w2" in coord.connections

    def test_consensus_majority(self):
        swarm = Swarm(config=SwarmConfig(consensus_mechanism=ConsensusMechanism.MAJORITY))
        swarm.add_agent(agent_id="a1")
        swarm.add_agent(agent_id="a2")
        swarm.add_agent(agent_id="a3")
        proposal_id = swarm.request_consensus("p1", "Use Python")
        swarm.vote("p1", "a1", "Python")
        swarm.vote("p1", "a2", "Python")
        swarm.vote("p1", "a3", "Rust")
        result = swarm.get_consensus_result("p1")
        assert result == "Python"

    def test_consensus_unanimous(self):
        swarm = Swarm(config=SwarmConfig(consensus_mechanism=ConsensusMechanism.UNANIMOUS))
        swarm.add_agent(agent_id="a1")
        swarm.add_agent(agent_id="a2")
        proposal_id = swarm.request_consensus("p1", "Deploy")
        swarm.vote("p1", "a1", "yes")
        swarm.vote("p1", "a2", "yes")
        result = swarm.get_consensus_result("p1")
        assert result == "yes"

    def test_status(self):
        swarm = Swarm()
        swarm.add_agent(agent_id="a1")
        status = swarm.status()
        assert status["swarm_id"] is not None
        assert status["agent_count"] == 1

    def test_get_topology(self):
        swarm = Swarm(config=SwarmConfig(topology=SwarmTopology.MESH))
        swarm.add_agent(agent_id="a1")
        swarm.add_agent(agent_id="a2")
        topo = swarm.get_topology()
        assert topo["topology"] == "mesh"
        assert len(topo["nodes"]) == 2
        assert len(topo["edges"]) >= 2


class TestSwarmManager:
    def test_create_swarm(self):
        manager = SwarmManager()
        swarm = manager.create_swarm()
        assert swarm.swarm_id is not None
        assert manager.get_swarm(swarm.swarm_id) is swarm

    def test_list_swarms(self):
        manager = SwarmManager()
        manager.create_swarm(swarm_id="s1")
        manager.create_swarm(swarm_id="s2")
        swarms = manager.list_swarms()
        assert len(swarms) == 2

    def test_destroy_swarm(self):
        manager = SwarmManager()
        swarm = manager.create_swarm(swarm_id="s1")
        assert manager.destroy_swarm("s1") is True
        assert manager.get_swarm("s1") is None

    def test_stats(self):
        manager = SwarmManager()
        manager.create_swarm(swarm_id="s1")
        stats = manager.stats()
        assert stats["total_swarms"] == 1
