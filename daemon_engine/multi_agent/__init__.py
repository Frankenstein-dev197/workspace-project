"""Multi-agent orchestration and swarm coordination system."""

from daemon_engine.multi_agent.orchestrator import Orchestrator
from daemon_engine.multi_agent.agent_manager import AgentManager
from daemon_engine.multi_agent.communication_system import CommunicationSystem, Message
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

__all__ = [
    "Orchestrator",
    "AgentManager",
    "CommunicationSystem",
    "Message",
    "Swarm",
    "SwarmManager",
    "SwarmConfig",
    "SwarmTopology",
    "SwarmAgent",
    "SwarmAgentRole",
    "SwarmAgentStatus",
    "ConsensusMechanism",
    "FailureHandling",
]
