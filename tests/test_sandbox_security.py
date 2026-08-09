"""Tests for sandbox security gating."""

import pytest

from daemon_engine.runtime.sandbox_security import (
    SecurityGate,
    SandboxSecurityConfig,
    SecurityResult,
    SecurityDecision,
    is_host_bash_allowed,
    uses_local_sandbox_provider,
    LOCAL_HOST_BASH_DISABLED_MESSAGE,
    LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE,
)


class TestSandboxSecurityConfig:
    def test_defaults(self):
        config = SandboxSecurityConfig()
        assert config.sandbox_use == ""
        assert config.allow_host_bash is False
        assert config.allow_privileged is False

    def test_is_local_provider_default(self):
        config = SandboxSecurityConfig()
        assert config.is_local_provider is False

    def test_is_local_provider_marker(self):
        config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.local:LocalSandboxProvider"
        )
        assert config.is_local_provider is True

    def test_is_local_provider_suffix(self):
        config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.local.submodule:LocalSandboxProvider"
        )
        assert config.is_local_provider is True

    def test_is_local_provider_wrong_module(self):
        config = SandboxSecurityConfig(
            sandbox_use="some.other.module:LocalSandboxProvider"
        )
        assert config.is_local_provider is False

    def test_is_local_provider_non_local(self):
        config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.aio:AioSandboxProvider"
        )
        assert config.is_local_provider is False


class TestUsesLocalSandboxProvider:
    def test_local_marker(self):
        config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.local:LocalSandboxProvider"
        )
        assert uses_local_sandbox_provider(config) is True

    def test_non_local(self):
        config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.docker:DockerSandboxProvider"
        )
        assert uses_local_sandbox_provider(config) is False


class TestIsHostBashAllowed:
    def test_non_local_provider_allowed(self):
        config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.aio:AioSandboxProvider"
        )
        assert is_host_bash_allowed(config) is True

    def test_local_provider_default_denied(self):
        config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.local:LocalSandboxProvider"
        )
        assert is_host_bash_allowed(config) is False

    def test_local_provider_explicit_allow(self):
        config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.local:LocalSandboxProvider",
            allow_host_bash=True,
        )
        assert is_host_bash_allowed(config) is True


class TestSecurityResult:
    def test_allow(self):
        result = SecurityResult(decision=SecurityDecision.ALLOW)
        assert result.allowed is True
        assert result.denied is False

    def test_deny(self):
        result = SecurityResult(decision=SecurityDecision.DENY, reason="bad")
        assert result.allowed is False
        assert result.denied is True
        assert result.reason == "bad"

    def test_to_dict(self):
        result = SecurityResult(
            decision=SecurityDecision.DENY,
            reason="blocked",
            command="rm -rf /",
        )
        d = result.to_dict()
        assert d["decision"] == "deny"
        assert d["reason"] == "blocked"
        assert d["command"] == "rm -rf /"

    def test_frozen(self):
        result = SecurityResult(decision=SecurityDecision.ALLOW)
        with pytest.raises(Exception):
            result.decision = SecurityDecision.DENY  # type: ignore[misc]


class TestSecurityGateDefaults:
    def test_default_config(self):
        gate = SecurityGate()
        assert gate.config.allow_host_bash is False

    def test_custom_config(self):
        config = SandboxSecurityConfig(allow_host_bash=True)
        gate = SecurityGate(config)
        assert gate.config.allow_host_bash is True


class TestSecurityGateCheckCommand:
    def setup_method(self):
        self.gate = SecurityGate(
            SandboxSecurityConfig(
                sandbox_use="daemon_engine.runtime.aio:AioSandboxProvider"
            )
        )

    def test_safe_command_allowed(self):
        result = self.gate.check_command("ls -la")
        assert result.allowed is True

    def test_empty_command_denied(self):
        result = self.gate.check_command("")
        assert result.denied is True
        assert "Empty" in result.reason

    def test_whitespace_command_denied(self):
        result = self.gate.check_command("   ")
        assert result.denied is True

    def test_rm_rf_root_denied(self):
        result = self.gate.check_command("rm -rf /")
        assert result.denied is True
        assert "Blocked" in result.reason

    def test_rm_rf_root_with_path_denied(self):
        result = self.gate.check_command("sudo rm -rf /home")
        assert result.denied is True

    def test_mkfs_denied(self):
        result = self.gate.check_command("mkfs.ext4 /dev/sda1")
        assert result.denied is True

    def test_fork_bomb_denied(self):
        result = self.gate.check_command(":(){ :|:& };:")
        assert result.denied is True

    def test_shutdown_denied(self):
        result = self.gate.check_command("shutdown -h now")
        assert result.denied is True

    def test_sudo_denied_by_default(self):
        result = self.gate.check_command("sudo apt update")
        assert result.denied is True
        assert "Privileged" in result.reason

    def test_sudo_allowed_with_flag(self):
        gate = SecurityGate(
            SandboxSecurityConfig(
                sandbox_use="daemon_engine.runtime.aio:AioSandboxProvider",
                allow_privileged=True,
            )
        )
        result = gate.check_command("sudo apt update")
        assert result.allowed is True

    def test_su_denied(self):
        result = self.gate.check_command("su root")
        assert result.denied is True

    def test_mount_denied(self):
        result = self.gate.check_command("mount /dev/sda /mnt")
        assert result.denied is True

    def test_case_insensitive_blocked(self):
        result = self.gate.check_command("RM -RF /")
        assert result.denied is True

    def test_is_safe_command(self):
        assert self.gate.is_safe_command("ls") is True
        assert self.gate.is_safe_command("rm -rf /") is False


class TestSecurityGateLocalSandbox:
    def setup_method(self):
        self.config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.local:LocalSandboxProvider"
        )
        self.gate = SecurityGate(self.config)

    def test_host_bash_denied_by_default(self):
        result = self.gate.check_command("ls")
        assert result.denied is True
        assert LOCAL_HOST_BASH_DISABLED_MESSAGE in result.reason

    def test_host_bash_allowed_with_flag(self):
        config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.local:LocalSandboxProvider",
            allow_host_bash=True,
        )
        gate = SecurityGate(config)
        result = gate.check_command("ls")
        assert result.allowed is True

    def test_check_host_bash_denied(self):
        result = self.gate.check_host_bash()
        assert result.denied is True
        assert LOCAL_HOST_BASH_DISABLED_MESSAGE in result.reason

    def test_check_host_bash_allowed(self):
        config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.local:LocalSandboxProvider",
            allow_host_bash=True,
        )
        gate = SecurityGate(config)
        result = gate.check_host_bash()
        assert result.allowed is True

    def test_check_bash_subagent_denied(self):
        result = self.gate.check_bash_subagent()
        assert result.denied is True
        assert LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE in result.reason

    def test_check_bash_subagent_allowed(self):
        config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.local:LocalSandboxProvider",
            allow_host_bash=True,
        )
        gate = SecurityGate(config)
        result = gate.check_bash_subagent()
        assert result.allowed is True

    def test_blocked_command_denied_even_with_allow(self):
        config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.local:LocalSandboxProvider",
            allow_host_bash=True,
        )
        gate = SecurityGate(config)
        result = gate.check_command("rm -rf /")
        assert result.denied is True


class TestSecurityGateNonLocalSandbox:
    def setup_method(self):
        self.config = SandboxSecurityConfig(
            sandbox_use="daemon_engine.runtime.aio:AioSandboxProvider"
        )
        self.gate = SecurityGate(self.config)

    def test_check_host_bash_allowed(self):
        result = self.gate.check_host_bash()
        assert result.allowed is True

    def test_check_bash_subagent_allowed(self):
        result = self.gate.check_bash_subagent()
        assert result.allowed is True

    def test_safe_command_allowed(self):
        result = self.gate.check_command("echo hello")
        assert result.allowed is True
