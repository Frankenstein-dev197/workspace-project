"""Swarm coordination: multi-agent topology and consensus management.

Integrates Ruflo's swarm system (hierarchical, mesh, adaptive topologies)
with consensus mechanisms (majority, unanimous, weighted). Enables agents
to work as coordinated swarms rather than isolated workers.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class SwarmTopology(str, Enum):
    """Swarm coordination topologies (from Ruflo)."""
    HIERARCHICAL = "hierarchical"
    MESH = "mesh"
    ADAPTIVE = "adaptive"
    COLLECTIVE = "collective"
    HIERARCHICAL_MESH = "hierarchical-mesh"


class ConsensusMechanism(str, Enum):
    """How the swarm reaches decisions."""
    MAJORITY = "majority"
    UNANIMOUS = "unanimous"
    WEIGHTED = "weighted"
    NONE = "none"


class FailureHandling(str, Enum):
    """How to handle agent failures."""
    RETRY = "retry"
    FAILOVER = "failover"
    IGNORE = "ignore"


class SwarmAgentRole(str, Enum):
    """Agent roles within a swarm."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    SPECIALIST = "specialist"


class SwarmAgentStatus(str, Enum):
    """Agent status within a swarm."""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    TERMINATED = "terminated"


@dataclass
class SwarmAgent:
    """An agent within a swarm."""
    id: str
    type: str = "worker"
    role: SwarmAgentRole = SwarmAgentRole.WORKER
    status: SwarmAgentStatus = SwarmAgentStatus.IDLE
    connections: list[str] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    weight: float = 1.0
    current_task: str = ""
    tasks_completed: int = 0
    tasks_failed: int = 0
    joined_at: float = field(default_factory=time.time)

    def is_available(self) -> bool:
        return self.status in (SwarmAgentStatus.IDLE, SwarmAgentStatus.ACTIVE)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "role": self.role.value,
            "status": self.status.value,
            "connections": self.connections,
            "capabilities": self.capabilities,
            "weight": self.weight,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
        }


@dataclass
class SwarmConfig:
    """Configuration for a swarm."""
    topology: SwarmTopology = SwarmTopology.HIERARCHICAL_MESH
    max_agents: int = 15
    communication_protocol: str = "message-bus"
    consensus_mechanism: ConsensusMechanism = ConsensusMechanism.MAJORITY
    failure_handling: FailureHandling = FailureHandling.RETRY
    load_balancing: bool = True
    auto_scaling: bool = False


@dataclass
class SwarmMetrics:
    """Performance metrics for a swarm."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    in_progress_tasks: int = 0
    average_task_duration: float = 0.0
    throughput: float = 0.0
    efficiency: float = 0.0
    uptime: float = 0.0


class Swarm:
    """A coordinated swarm of agents.

    Manages agent registration, task distribution, consensus voting,
    and topology-based communication routing.
    """

    def __init__(
        self,
        swarm_id: str | None = None,
        config: SwarmConfig | None = None,
    ) -> None:
        self.swarm_id = swarm_id or f"swarm-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
        self.config = config or SwarmConfig()
        self._agents: dict[str, SwarmAgent] = {}
        self._coordinator_id: str | None = None
        self._created_at = time.time()
        self._metrics = SwarmMetrics()
        self._task_queue: list[dict[str, Any]] = []
        self._consensus_votes: dict[str, dict[str, Any]] = {}

    @property
    def agent_count(self) -> int:
        return len(self._agents)

    @property
    def active_count(self) -> int:
        return sum(1 for a in self._agents.values() if a.status == SwarmAgentStatus.ACTIVE)

    def add_agent(
        self,
        agent_id: str | None = None,
        agent_type: str = "worker",
        role: SwarmAgentRole = SwarmAgentRole.WORKER,
        capabilities: list[str] | None = None,
        weight: float = 1.0,
    ) -> SwarmAgent:
        agent_id = agent_id or f"agent-{len(self._agents) + 1:04d}"
        if agent_id in self._agents:
            return self._agents[agent_id]
        agent = SwarmAgent(
            id=agent_id,
            type=agent_type,
            role=role,
            capabilities=capabilities or [],
            weight=weight,
        )
        self._agents[agent_id] = agent
        if role == SwarmAgentRole.COORDINATOR and not self._coordinator_id:
            self._coordinator_id = agent_id
        self._update_topology()
        logger.info("Swarm %s: added agent %s (%s)", self.swarm_id, agent_id, agent_type)
        return agent

    def remove_agent(self, agent_id: str) -> bool:
        agent = self._agents.pop(agent_id, None)
        if not agent:
            return False
        if self._coordinator_id == agent_id:
            coordinators = [a for a in self._agents.values() if a.role == SwarmAgentRole.COORDINATOR]
            self._coordinator_id = coordinators[0].id if coordinators else None
        for other in self._agents.values():
            if agent_id in other.connections:
                other.connections.remove(agent_id)
        self._update_topology()
        return True

    def get_agent(self, agent_id: str) -> SwarmAgent | None:
        return self._agents.get(agent_id)

    def list_agents(self, status: SwarmAgentStatus | None = None) -> list[SwarmAgent]:
        if status:
            return [a for a in self._agents.values() if a.status == status]
        return list(self._agents.values())

    def get_available_agent(self, required_capability: str | None = None) -> SwarmAgent | None:
        candidates = [a for a in self._agents.values() if a.is_available()]
        if required_capability:
            candidates = [a for a in candidates if required_capability in a.capabilities]
        if not candidates:
            return None
        if self.config.load_balancing:
            candidates.sort(key=lambda a: a.tasks_completed)
        return candidates[0]

    def assign_task(self, agent_id: str, task_id: str, task_data: dict[str, Any] | None = None) -> bool:
        agent = self._agents.get(agent_id)
        if not agent or not agent.is_available():
            return False
        agent.status = SwarmAgentStatus.BUSY
        agent.current_task = task_id
        self._metrics.in_progress_tasks += 1
        self._metrics.total_tasks += 1
        logger.info("Swarm %s: assigned task %s to agent %s", self.swarm_id, task_id, agent_id)
        return True

    def complete_task(self, agent_id: str, task_id: str, success: bool = True) -> None:
        agent = self._agents.get(agent_id)
        if not agent:
            return
        agent.current_task = ""
        agent.status = SwarmAgentStatus.IDLE
        self._metrics.in_progress_tasks = max(0, self._metrics.in_progress_tasks - 1)
        if success:
            agent.tasks_completed += 1
            self._metrics.completed_tasks += 1
        else:
            agent.tasks_failed += 1
            self._metrics.failed_tasks += 1

    def request_consensus(self, proposal_id: str, proposal: str, voters: list[str] | None = None) -> str:
        voter_ids = voters or [a.id for a in self._agents.values() if a.status != SwarmAgentStatus.TERMINATED]
        self._consensus_votes[proposal_id] = {
            "proposal": proposal,
            "voters": voter_ids,
            "votes": {},
            "created_at": time.time(),
        }
        return proposal_id

    def vote(self, proposal_id: str, agent_id: str, vote_value: Any) -> bool:
        proposal = self._consensus_votes.get(proposal_id)
        if not proposal or agent_id not in proposal["voters"]:
            return False
        proposal["votes"][agent_id] = vote_value
        if len(proposal["votes"]) >= len(proposal["voters"]):
            return self._resolve_consensus(proposal_id) is not None
        return True

    def _resolve_consensus(self, proposal_id: str) -> Any | None:
        proposal = self._consensus_votes.get(proposal_id)
        if not proposal:
            return None
        votes = list(proposal["votes"].values())
        mechanism = self.config.consensus_mechanism
        if mechanism == ConsensusMechanism.UNANIMOUS:
            result = votes[0] if len(set(votes)) == 1 else None
        elif mechanism == ConsensusMechanism.MAJORITY:
            from collections import Counter
            counts = Counter(votes)
            result = counts.most_common(1)[0][0] if counts else None
        elif mechanism == ConsensusMechanism.WEIGHTED:
            weighted: dict[Any, float] = {}
            for agent_id, vote in proposal["votes"].items():
                agent = self._agents.get(agent_id)
                weight = agent.weight if agent else 1.0
                weighted[vote] = weighted.get(vote, 0) + weight
            result = max(weighted, key=weighted.get) if weighted else None
        else:
            result = votes[0] if votes else None
        proposal["result"] = result
        proposal["resolved_at"] = time.time()
        return result

    def get_consensus_result(self, proposal_id: str) -> Any | None:
        proposal = self._consensus_votes.get(proposal_id)
        if not proposal:
            return None
        return proposal.get("result")

    def _update_topology(self) -> None:
        topology = self.config.topology
        if topology == SwarmTopology.HIERARCHICAL:
            if self._coordinator_id:
                coordinator = self._agents.get(self._coordinator_id)
                if coordinator:
                    coordinator.connections = [
                        a.id for a in self._agents.values()
                        if a.id != self._coordinator_id and a.role != SwarmAgentRole.COORDINATOR
                    ]
                for agent in self._agents.values():
                    if agent.id != self._coordinator_id and agent.role != SwarmAgentRole.COORDINATOR:
                        agent.connections = [self._coordinator_id]
        elif topology == SwarmTopology.MESH:
            for agent in self._agents.values():
                agent.connections = [
                    a.id for a in self._agents.values() if a.id != agent.id
                ]
        elif topology == SwarmTopology.HIERARCHICAL_MESH:
            coordinators = [a for a in self._agents.values() if a.role == SwarmAgentRole.COORDINATOR]
            workers = [a for a in self._agents.values() if a.role != SwarmAgentRole.COORDINATOR]
            for coord in coordinators:
                coord.connections = [c.id for c in coordinators if c.id != coord.id] + [w.id for w in workers]
            for worker in workers:
                worker.connections = [c.id for c in coordinators] + [
                    w.id for w in workers if w.id != worker.id
                ][:3]
        elif topology == SwarmTopology.ADAPTIVE:
            for agent in self._agents.values():
                available = [a.id for a in self._agents.values() if a.id != agent.id and a.is_available()]
                agent.connections = available[:5]

    def scale(self, target_agents: int) -> int:
        current = len(self._agents)
        if target_agents > current:
            for _ in range(target_agents - current):
                if len(self._agents) >= self.config.max_agents:
                    break
                self.add_agent()
        elif target_agents < current:
            removable = [a.id for a in self._agents.values() if a.status == SwarmAgentStatus.IDLE]
            for agent_id in removable[: current - target_agents]:
                self.remove_agent(agent_id)
        self._update_topology()
        return len(self._agents)

    def get_topology(self) -> dict[str, Any]:
        return {
            "topology": self.config.topology.value,
            "nodes": [
                {"id": a.id, "type": a.type, "role": a.role.value}
                for a in self._agents.values()
            ],
            "edges": [
                {"from": a.id, "to": conn}
                for a in self._agents.values()
                for conn in a.connections
            ],
            "coordinator": self._coordinator_id,
        }

    def metrics(self) -> SwarmMetrics:
        self._metrics.uptime = time.time() - self._created_at
        if self._metrics.total_tasks > 0:
            self._metrics.efficiency = self._metrics.completed_tasks / self._metrics.total_tasks
        if self._metrics.uptime > 0:
            self._metrics.throughput = self._metrics.completed_tasks / self._metrics.uptime
        return self._metrics

    def status(self) -> dict[str, Any]:
        m = self.metrics()
        return {
            "swarm_id": self.swarm_id,
            "topology": self.config.topology.value,
            "agent_count": self.agent_count,
            "active_count": self.active_count,
            "coordinator": self._coordinator_id,
            "consensus_mechanism": self.config.consensus_mechanism.value,
            "metrics": {
                "total_tasks": m.total_tasks,
                "completed_tasks": m.completed_tasks,
                "failed_tasks": m.failed_tasks,
                "in_progress": m.in_progress_tasks,
                "efficiency": round(m.efficiency, 4),
                "uptime": round(m.uptime, 2),
            },
            "agents": [a.to_dict() for a in self._agents.values()],
        }


class SwarmManager:
    """Manages multiple swarms for different tasks/projects."""

    def __init__(self) -> None:
        self._swarms: dict[str, Swarm] = {}

    def create_swarm(
        self,
        config: SwarmConfig | None = None,
        swarm_id: str | None = None,
    ) -> Swarm:
        swarm = Swarm(swarm_id=swarm_id, config=config)
        self._swarms[swarm.swarm_id] = swarm
        logger.info("Created swarm %s with topology %s", swarm.swarm_id, swarm.config.topology.value)
        return swarm

    def get_swarm(self, swarm_id: str) -> Swarm | None:
        return self._swarms.get(swarm_id)

    def list_swarms(self) -> list[dict[str, Any]]:
        return [s.status() for s in self._swarms.values()]

    def destroy_swarm(self, swarm_id: str) -> bool:
        swarm = self._swarms.pop(swarm_id, None)
        return swarm is not None

    def stats(self) -> dict[str, Any]:
        return {
            "total_swarms": len(self._swarms),
            "total_agents": sum(s.agent_count for s in self._swarms.values()),
            "active_agents": sum(s.active_count for s in self._swarms.values()),
        }
