"""Sandbox security: capability gating for host bash and dangerous operations.

Integrates DeerFlow sandbox security pattern:
- SecurityGate: enforces sandbox capability requirements
  - host_bash: whether host bash execution is allowed
  - local_sandbox: whether the provider is a host-local (insecure) boundary
  - fail-closed: dangerous operations blocked by default
- is_host_bash_allowed: checks if host bash is permitted for the provider
- uses_local_sandbox_provider: detects insecure local sandbox markers
- SecurityDecision: allow/deny with reason
- Command safety checking: blocked commands, path restrictions

Local sandbox providers run commands directly on the host without
isolation, so host bash is disabled by default for them. Only
explicit opt-in (allow_host_bash=True) enables it, and only in
trusted local environments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

LOCAL_SANDBOX_PROVIDER_MARKERS = (
    "daemon_engine.runtime.local:LocalSandboxProvider",
    "daemon_engine.runtime.local.local_sandbox_provider:LocalSandboxProvider",
)

LOCAL_HOST_BASH_DISABLED_MESSAGE = (
    "Host bash execution is disabled for LocalSandboxProvider because it is not "
    "a secure sandbox boundary. Switch to an isolated sandbox provider, or set "
    "allow_host_bash=True only in a fully trusted local environment."
)

LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE = (
    "Bash subagent is disabled for LocalSandboxProvider because host bash "
    "execution is not a secure sandbox boundary. Switch to an isolated sandbox "
    "provider, or set allow_host_bash=True only in a fully trusted local environment."
)

# Commands that are always blocked regardless of configuration
_BLOCKED_COMMANDS = frozenset({
    "rm -rf /",
    "mkfs",
    "dd if=/dev/zero of=/dev/sda",
    ":(){ :|:& };:",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
})

# Commands that require elevated privileges — blocked in sandbox
_PRIVILEGED_COMMANDS = frozenset({
    "sudo",
    "su",
    "chmod 777",
    "chown root",
    "mount",
    "umount",
    "iptables",
    "nsenter",
    "setcap",
})


class SecurityDecision(Enum):
    """Result of a security check."""
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True)
class SecurityResult:
    """Result of a security gate check."""
    decision: SecurityDecision
    reason: str = ""
    command: str = ""

    @property
    def allowed(self) -> bool:
        return self.decision is SecurityDecision.ALLOW

    @property
    def denied(self) -> bool:
        return self.decision is SecurityDecision.DENY

    def to_dict(self) -> dict[str, str]:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "command": self.command,
        }


class SandboxSecurityConfig:
    """Configuration for sandbox security gating."""

    def __init__(
        self,
        sandbox_use: str = "",
        allow_host_bash: bool = False,
        allow_privileged: bool = False,
    ) -> None:
        self.sandbox_use = sandbox_use
        self.allow_host_bash = allow_host_bash
        self.allow_privileged = allow_privileged

    @property
    def is_local_provider(self) -> bool:
        """Check if the configured sandbox provider is a local (host) provider."""
        if self.sandbox_use in LOCAL_SANDBOX_PROVIDER_MARKERS:
            return True
        return (
            self.sandbox_use.endswith(":LocalSandboxProvider")
            and "daemon_engine.runtime.local" in self.sandbox_use
        )


def uses_local_sandbox_provider(config: SandboxSecurityConfig) -> bool:
    """Return True when the active sandbox provider is the host-local provider."""
    return config.is_local_provider


def is_host_bash_allowed(config: SandboxSecurityConfig) -> bool:
    """Return whether host bash execution is explicitly allowed."""
    if not uses_local_sandbox_provider(config):
        return True
    return config.allow_host_bash


class SecurityGate:
    """Enforces sandbox capability requirements for commands.

    Fail-closed: dangerous operations are blocked by default unless
    explicitly allowed by configuration.
    """

    def __init__(self, config: SandboxSecurityConfig | None = None) -> None:
        self.config = config or SandboxSecurityConfig()

    def check_command(self, command: str) -> SecurityResult:
        """Check if a command is allowed to execute.

        Returns SecurityResult with decision and reason.
        """
        cmd = command.strip()
        if not cmd:
            return SecurityResult(
                decision=SecurityDecision.DENY,
                reason="Empty command",
                command=command,
            )

        # Check blocked commands
        cmd_lower = cmd.lower()
        for blocked in _BLOCKED_COMMANDS:
            if blocked in cmd_lower:
                return SecurityResult(
                    decision=SecurityDecision.DENY,
                    reason=f"Blocked command pattern: {blocked}",
                    command=command,
                )

        # Check privileged commands
        if not self.config.allow_privileged:
            for priv in _PRIVILEGED_COMMANDS:
                if priv in cmd_lower:
                    return SecurityResult(
                        decision=SecurityDecision.DENY,
                        reason=f"Privileged command not allowed: {priv}",
                        command=command,
                    )

        # Check host bash for local sandbox
        if uses_local_sandbox_provider(self.config):
            if not is_host_bash_allowed(self.config):
                return SecurityResult(
                    decision=SecurityDecision.DENY,
                    reason=LOCAL_HOST_BASH_DISABLED_MESSAGE,
                    command=command,
                )

        return SecurityResult(
            decision=SecurityDecision.ALLOW,
            command=command,
        )

    def check_host_bash(self) -> SecurityResult:
        """Check if host bash is allowed at all."""
        if uses_local_sandbox_provider(self.config):
            if not is_host_bash_allowed(self.config):
                return SecurityResult(
                    decision=SecurityDecision.DENY,
                    reason=LOCAL_HOST_BASH_DISABLED_MESSAGE,
                )
        return SecurityResult(decision=SecurityDecision.ALLOW)

    def check_bash_subagent(self) -> SecurityResult:
        """Check if bash subagent is allowed."""
        if uses_local_sandbox_provider(self.config):
            if not is_host_bash_allowed(self.config):
                return SecurityResult(
                    decision=SecurityDecision.DENY,
                    reason=LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE,
                )
        return SecurityResult(decision=SecurityDecision.ALLOW)

    def is_safe_command(self, command: str) -> bool:
        """Quick boolean check if a command is safe."""
        return self.check_command(command).allowed
