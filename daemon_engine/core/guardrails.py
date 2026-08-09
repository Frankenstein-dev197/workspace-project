"""Guardrails: pre-tool-call authorization system.

Integrates DeerFlow's guardrails pattern: pluggable providers that
evaluate tool calls before execution, returning allow/deny decisions
with structured reasons. Supports allowlist/denylist, rate limiting,
and fail-closed behavior.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class GuardrailResult(str, Enum):
    """Result of a guardrail evaluation."""
    ALLOW = "allow"
    DENY = "deny"
    CHALLENGE = "challenge"


@dataclass
class GuardrailReason:
    """Structured reason for an allow/deny decision."""
    code: str
    message: str = ""


@dataclass
class GuardrailRequest:
    """Context passed to the provider for each tool call."""
    tool_name: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    agent_id: str | None = None
    thread_id: str | None = None
    is_subagent: bool = False
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    user_id: str | None = None
    user_role: str | None = None
    run_id: str | None = None
    tool_call_id: str | None = None
    channel_user_id: str | None = None
    is_internal: bool = False
    authz_attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailDecision:
    """Provider's allow/deny verdict."""
    result: GuardrailResult
    reasons: list[GuardrailReason] = field(default_factory=list)
    policy_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def allow(self) -> bool:
        return self.result == GuardrailResult.ALLOW

    @property
    def deny(self) -> bool:
        return self.result == GuardrailResult.DENY


@runtime_checkable
class GuardrailProvider(Protocol):
    """Contract for pluggable tool-call authorization."""
    name: str

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision: ...


class AllowlistProvider:
    """Simple allowlist/denylist provider.

    None for allowed_tools means allow all (no allowlist configured).
    Empty list [] means allow nothing (explicitly configured).
    """

    name = "allowlist"

    def __init__(
        self,
        allowed_tools: list[str] | None = None,
        denied_tools: list[str] | None = None,
    ) -> None:
        self._allowed = set(allowed_tools) if allowed_tools is not None else None
        self._denied = set(denied_tools) if denied_tools else set()

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if self._allowed is not None and request.tool_name not in self._allowed:
            return GuardrailDecision(
                result=GuardrailResult.DENY,
                reasons=[GuardrailReason(
                    code="tool_not_allowed",
                    message=f"tool '{request.tool_name}' not in allowlist",
                )],
            )
        if request.tool_name in self._denied:
            return GuardrailDecision(
                result=GuardrailResult.DENY,
                reasons=[GuardrailReason(
                    code="tool_denied",
                    message=f"tool '{request.tool_name}' is denied",
                )],
            )
        return GuardrailDecision(
            result=GuardrailResult.ALLOW,
            reasons=[GuardrailReason(code="allowed")],
        )


class RateLimitProvider:
    """Rate limiting provider per tool or per agent."""

    name = "rate_limit"

    def __init__(
        self,
        max_calls_per_minute: int = 60,
        max_calls_per_hour: int = 1000,
        per_tool: bool = True,
    ) -> None:
        self._max_per_min = max_calls_per_minute
        self._max_per_hour = max_calls_per_hour
        self._per_tool = per_tool
        self._calls: dict[str, list[float]] = {}

    def _get_key(self, request: GuardrailRequest) -> str:
        if self._per_tool:
            return request.tool_name
        return request.agent_id or "global"

    def _prune_old(self, key: str, now: float) -> None:
        calls = self._calls.get(key, [])
        self._calls[key] = [t for t in calls if now - t < 3600]

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        key = self._get_key(request)
        now = time.time()
        self._prune_old(key, now)
        calls = self._calls.setdefault(key, [])
        recent_min = [t for t in calls if now - t < 60]
        recent_hour = [t for t in calls if now - t < 3600]
        if len(recent_min) >= self._max_per_min:
            return GuardrailDecision(
                result=GuardrailResult.DENY,
                reasons=[GuardrailReason(
                    code="rate_limit_exceeded",
                    message=f"per-minute limit ({self._max_per_min}) exceeded for {key}",
                )],
                metadata={"calls_in_last_minute": len(recent_min)},
            )
        if len(recent_hour) >= self._max_per_hour:
            return GuardrailDecision(
                result=GuardrailResult.DENY,
                reasons=[GuardrailReason(
                    code="rate_limit_exceeded",
                    message=f"per-hour limit ({self._max_per_hour}) exceeded for {key}",
                )],
                metadata={"calls_in_last_hour": len(recent_hour)},
            )
        calls.append(now)
        return GuardrailDecision(
            result=GuardrailResult.ALLOW,
            reasons=[GuardrailReason(code="within_rate_limit")],
        )


class InputValidationProvider:
    """Validate tool input against constraints."""

    name = "input_validation"

    def __init__(
        self,
        max_input_length: int = 10000,
        blocked_patterns: list[str] | None = None,
        required_fields: dict[str, list[str]] | None = None,
    ) -> None:
        self._max_length = max_input_length
        self._blocked_patterns = blocked_patterns or ["rm -rf", "sudo", "DROP TABLE"]
        self._required_fields = required_fields or {}

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        for field_name, value in request.tool_input.items():
            value_str = str(value)
            if len(value_str) > self._max_length:
                return GuardrailDecision(
                    result=GuardrailResult.DENY,
                    reasons=[GuardrailReason(
                        code="input_too_long",
                        message=f"field '{field_name}' exceeds {self._max_length} chars",
                    )],
                )
            for pattern in self._blocked_patterns:
                if pattern.lower() in value_str.lower():
                    return GuardrailDecision(
                        result=GuardrailResult.DENY,
                        reasons=[GuardrailReason(
                            code="blocked_pattern",
                            message=f"field '{field_name}' contains blocked pattern",
                        )],
                    )
        required = self._required_fields.get(request.tool_name, [])
        for field in required:
            if field not in request.tool_input:
                return GuardrailDecision(
                    result=GuardrailResult.DENY,
                    reasons=[GuardrailReason(
                        code="missing_required_field",
                        message=f"field '{field}' is required for {request.tool_name}",
                    )],
                )
        return GuardrailDecision(
            result=GuardrailResult.ALLOW,
            reasons=[GuardrailReason(code="input_valid")],
        )


class SubagentRestrictionProvider:
    """Restrict tools available to subagents."""

    name = "subagent_restriction"

    def __init__(self, subagent_denied_tools: list[str] | None = None) -> None:
        self._denied = set(subagent_denied_tools or ["task", "spawn_subagent"])

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if request.is_subagent and request.tool_name in self._denied:
            return GuardrailDecision(
                result=GuardrailResult.DENY,
                reasons=[GuardrailReason(
                    code="subagent_restricted",
                    message=f"subagents cannot use '{request.tool_name}'",
                )],
            )
        return GuardrailDecision(
            result=GuardrailResult.ALLOW,
            reasons=[GuardrailReason(code="not_restricted")],
        )


class GuardrailMiddleware:
    """Evaluate tool calls against providers before execution.

    Denied calls return an error message so the agent can adapt.
    If a provider raises, behavior depends on fail_closed:
    - True (default): block the call
    - False: allow it through with a warning
    """

    def __init__(
        self,
        providers: list[GuardrailProvider] | None = None,
        fail_closed: bool = True,
    ) -> None:
        self._providers: list[GuardrailProvider] = providers or []
        self._fail_closed = fail_closed
        self._stats = {
            "total_evaluations": 0,
            "allowed": 0,
            "denied": 0,
            "errors": 0,
            "by_provider": {},
        }

    def add_provider(self, provider: GuardrailProvider) -> None:
        self._providers.append(provider)

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """Evaluate a tool call against all providers."""
        self._stats["total_evaluations"] += 1
        all_reasons: list[GuardrailReason] = []
        for provider in self._providers:
            try:
                decision = provider.evaluate(request)
                provider_name = getattr(provider, "name", "unknown")
                self._stats["by_provider"].setdefault(provider_name, {"allow": 0, "deny": 0})
                if decision.deny:
                    self._stats["by_provider"][provider_name]["deny"] += 1
                    self._stats["denied"] += 1
                    return decision
                self._stats["by_provider"][provider_name]["allow"] += 1
                all_reasons.extend(decision.reasons)
            except Exception as exc:
                self._stats["errors"] += 1
                logger.error("Guardrail provider %s raised: %s", provider, exc)
                if self._fail_closed:
                    self._stats["denied"] += 1
                    return GuardrailDecision(
                        result=GuardrailResult.DENY,
                        reasons=[GuardrailReason(
                            code="provider_error",
                            message=f"Provider error: {exc}",
                        )],
                    )
        self._stats["allowed"] += 1
        return GuardrailDecision(
            result=GuardrailResult.ALLOW,
            reasons=all_reasons or [GuardrailReason(code="no_providers")],
        )

    def check_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        agent_id: str | None = None,
        is_subagent: bool = False,
    ) -> tuple[bool, str]:
        """Check a tool call. Returns (allowed, reason_message)."""
        request = GuardrailRequest(
            tool_name=tool_name,
            tool_input=tool_input,
            agent_id=agent_id,
            is_subagent=is_subagent,
        )
        decision = self.evaluate(request)
        if decision.deny:
            msg = "; ".join(r.message for r in decision.reasons)
            return False, msg
        return True, ""

    def stats(self) -> dict[str, Any]:
        return dict(self._stats)


def create_default_guardrails(
    allowed_tools: list[str] | None = None,
    denied_tools: list[str] | None = None,
    fail_closed: bool = True,
) -> GuardrailMiddleware:
    """Create a middleware with default providers."""
    providers: list[GuardrailProvider] = [
        AllowlistProvider(allowed_tools=allowed_tools, denied_tools=denied_tools),
        InputValidationProvider(),
        SubagentRestrictionProvider(),
    ]
    return GuardrailMiddleware(providers=providers, fail_closed=fail_closed)
