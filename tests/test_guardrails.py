"""Tests for guardrails system."""

import pytest

from daemon_engine.core.guardrails import (
    GuardrailMiddleware,
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


class TestGuardrailDecision:
    def test_allow(self):
        decision = GuardrailDecision(result=GuardrailResult.ALLOW)
        assert decision.allow is True
        assert decision.deny is False

    def test_deny(self):
        decision = GuardrailDecision(result=GuardrailResult.DENY)
        assert decision.allow is False
        assert decision.deny is True


class TestAllowlistProvider:
    def test_allow_all_by_default(self):
        provider = AllowlistProvider()
        req = GuardrailRequest(tool_name="any_tool")
        assert provider.evaluate(req).allow is True

    def test_allowlist_restricts(self):
        provider = AllowlistProvider(allowed_tools=["bash", "read"])
        req = GuardrailRequest(tool_name="write")
        assert provider.evaluate(req).deny is True

    def test_allowlist_allows_listed(self):
        provider = AllowlistProvider(allowed_tools=["bash"])
        req = GuardrailRequest(tool_name="bash")
        assert provider.evaluate(req).allow is True

    def test_empty_allowlist_blocks_all(self):
        provider = AllowlistProvider(allowed_tools=[])
        req = GuardrailRequest(tool_name="bash")
        assert provider.evaluate(req).deny is True

    def test_denylist_blocks(self):
        provider = AllowlistProvider(denied_tools=["rm"])
        req = GuardrailRequest(tool_name="rm")
        assert provider.evaluate(req).deny is True

    def test_denylist_allows_others(self):
        provider = AllowlistProvider(denied_tools=["rm"])
        req = GuardrailRequest(tool_name="bash")
        assert provider.evaluate(req).allow is True


class TestRateLimitProvider:
    def test_allows_under_limit(self):
        provider = RateLimitProvider(max_calls_per_minute=10)
        req = GuardrailRequest(tool_name="bash")
        assert provider.evaluate(req).allow is True

    def test_denies_over_per_minute_limit(self):
        provider = RateLimitProvider(max_calls_per_minute=2)
        req = GuardrailRequest(tool_name="bash")
        provider.evaluate(req)
        provider.evaluate(req)
        decision = provider.evaluate(req)
        assert decision.deny is True

    def test_per_tool_isolation(self):
        provider = RateLimitProvider(max_calls_per_minute=1)
        provider.evaluate(GuardrailRequest(tool_name="bash"))
        decision = provider.evaluate(GuardrailRequest(tool_name="read"))
        assert decision.allow is True


class TestInputValidationProvider:
    def test_allows_valid_input(self):
        provider = InputValidationProvider()
        req = GuardrailRequest(tool_name="bash", tool_input={"command": "ls"})
        assert provider.evaluate(req).allow is True

    def test_denies_too_long_input(self):
        provider = InputValidationProvider(max_input_length=10)
        req = GuardrailRequest(tool_name="bash", tool_input={"command": "x" * 100})
        assert provider.evaluate(req).deny is True

    def test_denies_blocked_pattern(self):
        provider = InputValidationProvider()
        req = GuardrailRequest(tool_name="bash", tool_input={"command": "rm -rf /"})
        assert provider.evaluate(req).deny is True

    def test_required_fields(self):
        provider = InputValidationProvider(required_fields={"write_file": ["path", "content"]})
        req = GuardrailRequest(tool_name="write_file", tool_input={"path": "test"})
        assert provider.evaluate(req).deny is True

    def test_required_fields_satisfied(self):
        provider = InputValidationProvider(required_fields={"write_file": ["path", "content"]})
        req = GuardrailRequest(tool_name="write_file", tool_input={"path": "test", "content": "data"})
        assert provider.evaluate(req).allow is True


class TestSubagentRestrictionProvider:
    def test_blocks_subagent_recursion(self):
        provider = SubagentRestrictionProvider()
        req = GuardrailRequest(tool_name="task", is_subagent=True)
        assert provider.evaluate(req).deny is True

    def test_allows_non_subagent(self):
        provider = SubagentRestrictionProvider()
        req = GuardrailRequest(tool_name="task", is_subagent=False)
        assert provider.evaluate(req).allow is True

    def test_allows_other_tools_for_subagent(self):
        provider = SubagentRestrictionProvider()
        req = GuardrailRequest(tool_name="bash", is_subagent=True)
        assert provider.evaluate(req).allow is True


class TestGuardrailMiddleware:
    def test_no_providers_allows(self):
        mw = GuardrailMiddleware(providers=[])
        req = GuardrailRequest(tool_name="bash")
        assert mw.evaluate(req).allow is True

    def test_provider_deny_blocks(self):
        mw = GuardrailMiddleware(providers=[AllowlistProvider(allowed_tools=["read"])])
        req = GuardrailRequest(tool_name="bash")
        assert mw.evaluate(req).deny is True

    def test_all_providers_allow(self):
        mw = GuardrailMiddleware(providers=[
            AllowlistProvider(allowed_tools=["bash"]),
            InputValidationProvider(),
        ])
        req = GuardrailRequest(tool_name="bash", tool_input={"cmd": "ls"})
        assert mw.evaluate(req).allow is True

    def test_first_deny_wins(self):
        mw = GuardrailMiddleware(providers=[
            AllowlistProvider(allowed_tools=[]),
            InputValidationProvider(),
        ])
        req = GuardrailRequest(tool_name="bash")
        assert mw.evaluate(req).deny is True

    def test_fail_closed_on_error(self):
        class ErrorProvider:
            name = "error"
            def evaluate(self, request):
                raise Exception("Provider error")

        mw = GuardrailMiddleware(providers=[ErrorProvider()], fail_closed=True)
        req = GuardrailRequest(tool_name="bash")
        assert mw.evaluate(req).deny is True

    def test_fail_open_on_error(self):
        class ErrorProvider:
            name = "error"
            def evaluate(self, request):
                raise Exception("Provider error")

        mw = GuardrailMiddleware(providers=[ErrorProvider()], fail_closed=False)
        req = GuardrailRequest(tool_name="bash")
        assert mw.evaluate(req).allow is True

    def test_check_tool_call(self):
        mw = GuardrailMiddleware(providers=[AllowlistProvider(allowed_tools=["bash"])])
        allowed, msg = mw.check_tool_call("bash", {})
        assert allowed is True
        allowed, msg = mw.check_tool_call("write", {})
        assert allowed is False
        assert msg

    def test_stats(self):
        mw = GuardrailMiddleware(providers=[AllowlistProvider(allowed_tools=["bash"])])
        mw.check_tool_call("bash", {})
        mw.check_tool_call("write", {})
        stats = mw.stats()
        assert stats["total_evaluations"] == 2
        assert stats["allowed"] == 1
        assert stats["denied"] == 1

    def test_add_provider(self):
        mw = GuardrailMiddleware()
        mw.add_provider(AllowlistProvider(allowed_tools=["bash"]))
        req = GuardrailRequest(tool_name="write")
        assert mw.evaluate(req).deny is True


class TestCreateDefaultGuardrails:
    def test_default_creation(self):
        mw = create_default_guardrails()
        req = GuardrailRequest(tool_name="bash", tool_input={"cmd": "ls"})
        assert mw.evaluate(req).allow is True

    def test_with_allowlist(self):
        mw = create_default_guardrails(allowed_tools=["bash"])
        req = GuardrailRequest(tool_name="write")
        assert mw.evaluate(req).deny is True

    def test_blocks_dangerous_input(self):
        mw = create_default_guardrails()
        req = GuardrailRequest(tool_name="bash", tool_input={"cmd": "rm -rf /"})
        assert mw.evaluate(req).deny is True

    def test_blocks_subagent_recursion(self):
        mw = create_default_guardrails()
        req = GuardrailRequest(tool_name="task", is_subagent=True)
        assert mw.evaluate(req).deny is True
