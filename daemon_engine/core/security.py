"""Security module: environment scrubbing and command safety.

Integrates DeerFlow's env_policy (secret scrubbing for sandbox subprocesses)
and learn-claude-code's dangerous command blocking. Ensures agents cannot
accidentally exfiltrate credentials or execute destructive commands.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


SECRET_NAME_PATTERNS: tuple[str, ...] = (
    "*KEY*",
    "*SECRET*",
    "*TOKEN*",
    "*PASS*",
    "*CREDENTIAL*",
    "*DSN*",
)

BLOCKED_EXACT_NAMES: frozenset[str] = frozenset({
    "DATABASE_URL",
    "DATABASE_URI",
    "REDIS_URL",
    "MONGODB_URI",
    "MONGO_URL",
    "AMQP_URL",
    "RABBITMQ_URL",
    "POSTGRES_URL",
    "POSTGRESQL_URL",
    "MYSQL_URL",
    "CLICKHOUSE_URL",
    "CONNECTION_STRING",
    "CONN_STR",
    "GH_PAT",
    "GITHUB_PAT",
    "MYSQL_PWD",
    "REDISCLI_AUTH",
    "REDIS_AUTH",
    "PGSERVICEFILE",
})

DANGEROUS_COMMAND_PATTERNS: tuple[str, ...] = (
    "rm -rf /",
    "rm -rf ~",
    "rm -rf $HOME",
    "sudo ",
    "shutdown",
    "reboot",
    "mkfs",
    "dd if=/dev/zero",
    "dd if=/dev/random",
    "> /dev/sda",
    "> /dev/nvme",
    ":(){:|:&};:",
    "fork bomb",
    "chmod -R 777 /",
    "chown -R",
    "kill -9 -1",
    "pkill -f python",
    "pkill -f server",
    "curl | bash",
    "curl | sh",
    "wget | bash",
    "wget | sh",
    "| bash",
    "| sh",
)

PATH_TRAVERSAL_PATTERNS: tuple[str, ...] = (
    "../",
    "..\\",
    "~/",
    "/etc/passwd",
    "/etc/shadow",
    "/root/",
    "/proc/self",
)


@dataclass
class SecurityCheckResult:
    """Result of a security check."""
    safe: bool
    reason: str = ""
    blocked_patterns: list[str] = field(default_factory=list)
    sanitized_value: str = ""


def is_blocked_env_name(name: str) -> bool:
    """Check if an environment variable name looks like a credential."""
    upper = name.upper()
    if upper in BLOCKED_EXACT_NAMES:
        return True
    return any(fnmatch.fnmatchcase(upper, pattern) for pattern in SECRET_NAME_PATTERNS)


def build_sandbox_env(injected: dict[str, str] | None = None) -> dict[str, str]:
    """Build a sanitized environment for sandbox subprocesses.

    Inherits os.environ minus secret-looking variables, then layers
    explicitly injected secrets on top. Injected secrets win because
    they are authorized by the skill declaration.
    """
    env = {
        key: value
        for key, value in os.environ.items()
        if not is_blocked_env_name(key)
    }
    if injected:
        env.update(injected)
    return env


def check_command_safety(command: str) -> SecurityCheckResult:
    """Check if a shell command is safe to execute."""
    blocked: list[str] = []
    for pattern in DANGEROUS_COMMAND_PATTERNS:
        if pattern in command:
            blocked.append(pattern)
    if blocked:
        return SecurityCheckResult(
            safe=False,
            reason=f"Blocked dangerous command pattern(s): {', '.join(blocked)}",
            blocked_patterns=blocked,
        )
    return SecurityCheckResult(safe=True)


def check_path_safety(path: str) -> SecurityCheckResult:
    """Check if a file path is safe (no traversal or sensitive system paths)."""
    blocked: list[str] = []
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if pattern in path:
            blocked.append(pattern)
    if blocked:
        return SecurityCheckResult(
            safe=False,
            reason=f"Blocked path pattern(s): {', '.join(blocked)}",
            blocked_patterns=blocked,
        )
    return SecurityCheckResult(safe=True)


def sanitize_output(output: str, max_length: int = 50000) -> str:
    """Sanitize command output: truncate and redact potential secrets."""
    if len(output) > max_length:
        output = output[:max_length] + "\n... [truncated by security module]"
    secret_patterns = [
        (re.compile(r"sk-[A-Za-z0-9]{20,}"), "sk-REDACTED"),
        (re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"), "ghp_REDACTED"),
        (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA_REDACTED"),
        (re.compile(r"Bearer\s+[A-Za-z0-9\-._~+\/]+=*"), "Bearer REDACTED"),
        (re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[=:]\s*\S+"), r"\1=REDACTED"),
    ]
    for pattern, replacement in secret_patterns:
        output = pattern.sub(replacement, output)
    return output


def validate_tool_input(tool_name: str, tool_input: dict[str, Any]) -> SecurityCheckResult:
    """Validate tool input parameters for security issues."""
    for key, value in tool_input.items():
        if isinstance(value, str):
            if key in ("command", "cmd", "script", "code"):
                result = check_command_safety(value)
                if not result.safe:
                    return result
            if key in ("path", "file", "filename", "filepath"):
                result = check_path_safety(value)
                if not result.safe:
                    return result
    return SecurityCheckResult(safe=True)


def scan_for_secrets(text: str) -> list[dict[str, str]]:
    """Scan text for potential secrets and return findings."""
    findings: list[dict[str, str]] = []
    patterns = [
        ("API Key", re.compile(r"(?i)(api[_-]?key)\s*[=:]\s*([A-Za-z0-9\-_]{20,})")),
        ("Bearer Token", re.compile(r"Bearer\s+([A-Za-z0-9\-._~+\/]+=*)")),
        ("OpenAI Key", re.compile(r"(sk-[A-Za-z0-9]{20,})")),
        ("GitHub Token", re.compile(r"(gh[pousr]_[A-Za-z0-9]{36,})")),
        ("AWS Key", re.compile(r"(AKIA[0-9A-Z]{16})")),
        ("Generic Secret", re.compile(r"(?i)(secret|password|token)\s*[=:]\s*(\S{8,})")),
    ]
    for name, pattern in patterns:
        for match in pattern.finditer(text):
            findings.append({
                "type": name,
                "value_preview": match.group(0)[:20] + "...",
                "position": str(match.start()),
            })
    return findings


class SecurityManager:
    """Central security manager for the daemon engine.

    Provides command safety checks, environment scrubbing, output sanitization,
    and secret scanning. Used by sandbox, code executor, and hooks system.
    """

    def __init__(self, strict_mode: bool = True) -> None:
        self.strict_mode = strict_mode
        self._blocked_commands: int = 0
        self._blocked_paths: int = 0
        self._scrubbed_envs: int = 0
        self._sanitized_outputs: int = 0
        self._secrets_found: int = 0

    def check_command(self, command: str) -> SecurityCheckResult:
        result = check_command_safety(command)
        if not result.safe:
            self._blocked_commands += 1
            logger.warning("Blocked command: %s", result.reason)
        return result

    def check_path(self, path: str) -> SecurityCheckResult:
        result = check_path_safety(path)
        if not result.safe:
            self._blocked_paths += 1
            logger.warning("Blocked path: %s", result.reason)
        return result

    def get_sandbox_env(self, injected: dict[str, str] | None = None) -> dict[str, str]:
        self._scrubbed_envs += 1
        return build_sandbox_env(injected)

    def sanitize(self, output: str, max_length: int = 50000) -> str:
        self._sanitized_outputs += 1
        return sanitize_output(output, max_length)

    def validate_tool_input(self, tool_name: str, tool_input: dict[str, Any]) -> SecurityCheckResult:
        return validate_tool_input(tool_name, tool_input)

    def scan_secrets(self, text: str) -> list[dict[str, str]]:
        findings = scan_for_secrets(text)
        self._secrets_found += len(findings)
        return findings

    def stats(self) -> dict[str, Any]:
        return {
            "strict_mode": self.strict_mode,
            "blocked_commands": self._blocked_commands,
            "blocked_paths": self._blocked_paths,
            "scrubbed_envs": self._scrubbed_envs,
            "sanitized_outputs": self._sanitized_outputs,
            "secrets_found": self._secrets_found,
            "dangerous_patterns": len(DANGEROUS_COMMAND_PATTERNS),
            "blocked_env_names": len(BLOCKED_EXACT_NAMES),
            "secret_patterns": len(SECRET_NAME_PATTERNS),
        }
