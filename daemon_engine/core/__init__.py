"""Core engine modules: agent, reasoning, planner, decision, hooks, messages."""

from daemon_engine.core.agent_engine import Agent, AgentEngine
from daemon_engine.core.reasoning_engine import ReasoningEngine
from daemon_engine.core.task_planner import TaskPlanner, Task, TaskStatus
from daemon_engine.core.decision_system import DecisionSystem
from daemon_engine.core.hooks import HookRegistry, HookEvent, HookContext, HookResult, create_default_registry
from daemon_engine.core.message_manager import MessageManager, Message, MessageRole, CompactionSettings
from daemon_engine.core.security import SecurityManager, SecurityCheckResult, build_sandbox_env, check_command_safety, sanitize_output
from daemon_engine.core.context_compact import ContextCompactor, auto_compact, snip_compact, micro_compact, compact_history
from daemon_engine.core.error_recovery import ErrorRecoveryManager, RecoveryResult, RecoveryAction, ErrorType, with_retry
from daemon_engine.core.task_graph import TaskGraph, GraphTask, TaskStatus as GraphTaskStatus
from daemon_engine.core.agent_brain import AgentBrain, AgentPlanner, AgentStep, PlanItem, PlanItemStatus
from daemon_engine.core.watchdog import Watchdog, ActionRecord, LoopDetection, LoopType
from daemon_engine.core.background_tasks import BackgroundTaskManager, BackgroundTask, TaskStatus as BgTaskStatus
from daemon_engine.core.system_prompt import SystemPromptBuilder, SkillRegistry, SkillEntry, create_prompt_builder
from daemon_engine.core.skill_catalog import SkillCatalog, build_catalog
from daemon_engine.core.dedupe_store import MemoryDedupeStore, DedupeStore, make_dedupe_key
from daemon_engine.core.mcp_plugin import (
    MCPClient,
    MCPCatalog,
    MCPToolDef,
    normalize_mcp_name,
    make_mcp_tool_name,
)
from daemon_engine.core.guardrails import (
    GuardrailMiddleware,
    GuardrailProvider,
    GuardrailRequest,
    GuardrailDecision,
    GuardrailReason,
    GuardrailResult,
    AllowlistProvider,
    RateLimitProvider,
    InputValidationProvider,
    SubagentRestrictionProvider,
    create_default_guardrails,
)

__all__ = [
    "Agent",
    "AgentEngine",
    "ReasoningEngine",
    "TaskPlanner",
    "Task",
    "TaskStatus",
    "DecisionSystem",
    "HookRegistry",
    "HookEvent",
    "HookContext",
    "HookResult",
    "create_default_registry",
    "MessageManager",
    "Message",
    "MessageRole",
    "CompactionSettings",
    "SecurityManager",
    "SecurityCheckResult",
    "build_sandbox_env",
    "check_command_safety",
    "sanitize_output",
    "ContextCompactor",
    "auto_compact",
    "snip_compact",
    "micro_compact",
    "compact_history",
    "ErrorRecoveryManager",
    "RecoveryResult",
    "RecoveryAction",
    "ErrorType",
    "with_retry",
    "TaskGraph",
    "GraphTask",
    "GraphTaskStatus",
    "GuardrailMiddleware",
    "GuardrailProvider",
    "GuardrailRequest",
    "GuardrailDecision",
    "GuardrailReason",
    "GuardrailResult",
    "AllowlistProvider",
    "RateLimitProvider",
    "InputValidationProvider",
    "SubagentRestrictionProvider",
    "create_default_guardrails",
    "AgentBrain",
    "AgentPlanner",
    "AgentStep",
    "PlanItem",
    "PlanItemStatus",
    "Watchdog",
    "ActionRecord",
    "LoopDetection",
    "LoopType",
    "BackgroundTaskManager",
    "BackgroundTask",
    "BgTaskStatus",
    "SystemPromptBuilder",
    "SkillRegistry",
    "SkillEntry",
    "create_prompt_builder",
    "SkillCatalog",
    "build_catalog",
    "MemoryDedupeStore",
    "DedupeStore",
    "make_dedupe_key",
    "MCPClient",
    "MCPCatalog",
    "MCPToolDef",
    "normalize_mcp_name",
    "make_mcp_tool_name",
]
