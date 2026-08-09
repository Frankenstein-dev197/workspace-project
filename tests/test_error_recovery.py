"""Tests for error recovery system."""

import pytest
import time

from daemon_engine.core.error_recovery import (
    ErrorRecoveryManager,
    ErrorType,
    RecoveryAction,
    RecoveryResult,
    RecoveryState,
    classify_error,
    compute_backoff_delay,
    determine_action,
    apply_action,
    with_retry,
    DEFAULT_MAX_TOKENS,
    ESCALATED_MAX_TOKENS,
    MAX_CONTINUATION_RETRIES,
    CONTINUATION_PROMPT,
)


class TestClassifyError:
    def test_max_tokens(self):
        err = Exception("max_tokens exceeded")
        assert classify_error(err) == ErrorType.MAX_TOKENS

    def test_prompt_too_long(self):
        err = Exception("prompt_too_long error")
        assert classify_error(err) == ErrorType.PROMPT_TOO_LONG

    def test_rate_limit(self):
        err = Exception("429 rate limit exceeded")
        assert classify_error(err) == ErrorType.RATE_LIMIT

    def test_overloaded(self):
        err = Exception("529 overloaded_error")
        assert classify_error(err) == ErrorType.OVERLOADED

    def test_timeout(self):
        err = Exception("Request timeout")
        assert classify_error(err) == ErrorType.TIMEOUT

    def test_connection(self):
        err = Exception("ConnectionError")
        assert classify_error(err) == ErrorType.CONNECTION

    def test_unknown(self):
        err = Exception("Something weird happened")
        assert classify_error(err) == ErrorType.UNKNOWN


class TestBackoff:
    def test_delay_increases(self):
        delay1 = compute_backoff_delay(0)
        delay2 = compute_backoff_delay(3)
        assert delay2 > delay1

    def test_max_delay(self):
        delay = compute_backoff_delay(20, max_delay_ms=1000)
        assert delay <= 1.1  # max + jitter

    def test_jitter(self):
        delays = [compute_backoff_delay(2) for _ in range(10)]
        assert len(set(delays)) > 1  # jitter adds variance


class TestDetermineAction:
    def test_max_tokens_escalate(self):
        state = RecoveryState()
        action = determine_action(ErrorType.MAX_TOKENS, state)
        assert action == RecoveryAction.ESCALATE_TOKENS

    def test_max_tokens_continuation(self):
        state = RecoveryState(token_escalated=True)
        action = determine_action(ErrorType.MAX_TOKENS, state)
        assert action == RecoveryAction.CONTINUATION

    def test_max_tokens_give_up(self):
        state = RecoveryState(token_escalated=True, continuation_count=MAX_CONTINUATION_RETRIES)
        action = determine_action(ErrorType.MAX_TOKENS, state)
        assert action == RecoveryAction.GIVE_UP

    def test_prompt_too_long_compact(self):
        state = RecoveryState()
        action = determine_action(ErrorType.PROMPT_TOO_LONG, state)
        assert action == RecoveryAction.COMPACT

    def test_prompt_too_long_give_up(self):
        state = RecoveryState(compacted=True)
        action = determine_action(ErrorType.PROMPT_TOO_LONG, state)
        assert action == RecoveryAction.GIVE_UP

    def test_rate_limit_backoff(self):
        state = RecoveryState()
        action = determine_action(ErrorType.RATE_LIMIT, state)
        assert action == RecoveryAction.BACKOFF

    def test_overloaded_fallback(self):
        state = RecoveryState(consecutive_529=3)
        action = determine_action(ErrorType.OVERLOADED, state)
        assert action == RecoveryAction.FALLBACK_MODEL

    def test_timeout_retry(self):
        state = RecoveryState(total_retries=1)
        action = determine_action(ErrorType.TIMEOUT, state)
        assert action == RecoveryAction.RETRY

    def test_timeout_give_up(self):
        state = RecoveryState(total_retries=100)
        action = determine_action(ErrorType.TIMEOUT, state)
        assert action == RecoveryAction.GIVE_UP


class TestApplyAction:
    def test_escalate_tokens(self):
        state = RecoveryState(current_max_tokens=DEFAULT_MAX_TOKENS)
        msgs, tokens, model = apply_action(
            RecoveryAction.ESCALATE_TOKENS, state, [], "", None
        )
        assert tokens == ESCALATED_MAX_TOKENS
        assert state.token_escalated is True

    def test_continuation(self):
        state = RecoveryState()
        msgs, tokens, model = apply_action(
            RecoveryAction.CONTINUATION, state, [{"role": "user", "content": "test"}], "", None
        )
        assert len(msgs) == 2
        assert CONTINUATION_PROMPT in msgs[-1]["content"]
        assert state.continuation_count == 1

    def test_compact(self):
        state = RecoveryState()
        compact_fn = lambda msgs: [{"role": "user", "content": "compacted"}]
        msgs, tokens, model = apply_action(
            RecoveryAction.COMPACT, state, [{"role": "user", "content": "long"}], "", compact_fn
        )
        assert msgs == [{"role": "user", "content": "compacted"}]
        assert state.compacted is True

    def test_fallback_model(self):
        state = RecoveryState(current_model="gpt-4")
        msgs, tokens, model = apply_action(
            RecoveryAction.FALLBACK_MODEL, state, [], "gpt-3.5", None
        )
        assert model == "gpt-3.5"
        assert state.model_switched is True


class TestWithRetry:
    def test_success_no_retry(self):
        def fn(msgs, tokens, model):
            return {"content": "success"}

        result = with_retry(fn, [{"role": "user", "content": "hi"}])
        assert result.success is True
        assert result.response == {"content": "success"}
        assert result.retries == 0

    def test_retries_on_error(self):
        attempts = [0]

        def fn(msgs, tokens, model):
            attempts[0] += 1
            if attempts[0] < 3:
                raise Exception("timeout error")
            return {"content": "success"}

        result = with_retry(fn, [{"role": "user", "content": "hi"}], max_retries=5)
        assert result.success is True
        assert result.retries == 2

    def test_give_up(self):
        def fn(msgs, tokens, model):
            raise Exception("Unknown catastrophic error")

        result = with_retry(fn, [{"role": "user", "content": "hi"}])
        assert result.success is False
        assert result.error_type == ErrorType.UNKNOWN

    def test_token_escalation(self):
        calls = []

        def fn(msgs, tokens, model):
            calls.append(tokens)
            if tokens == DEFAULT_MAX_TOKENS:
                raise Exception("max_tokens exceeded")
            return {"content": "success"}

        result = with_retry(fn, [{"role": "user", "content": "hi"}])
        assert result.success is True
        assert DEFAULT_MAX_TOKENS in calls
        assert ESCALATED_MAX_TOKENS in calls

    def test_compaction_triggered(self):
        compacted = [False]

        def fn(msgs, tokens, model):
            if not compacted[0]:
                raise Exception("prompt_too_long")
            return {"content": "success"}

        def compact_fn(msgs):
            compacted[0] = True
            return [{"role": "user", "content": "compacted"}]

        result = with_retry(fn, [{"role": "user", "content": "hi"}], compact_fn=compact_fn)
        assert result.success is True
        assert compacted[0] is True

    def test_model_fallback(self):
        models_used = []

        def fn(msgs, tokens, model):
            models_used.append(model)
            if model == "primary":
                raise Exception("529 overloaded")
            return {"content": "success"}

        # Simulate consecutive 529s
        call_count = [0]

        def fn2(msgs, tokens, model):
            call_count[0] += 1
            if call_count[0] <= 3:
                raise Exception("529 overloaded_error")
            return {"content": "success"}

        result = with_retry(
            fn2, [{"role": "user", "content": "hi"}],
            model="primary",
            fallback_model="secondary",
        )
        assert result.success is True


class TestErrorRecoveryManager:
    def test_execute_success(self):
        mgr = ErrorRecoveryManager()
        result = mgr.execute(
            lambda msgs, tokens, model: {"content": "ok"},
            [{"role": "user", "content": "hi"}],
        )
        assert result.success is True
        stats = mgr.stats()
        assert stats["total_calls"] == 1
        assert stats["successful_calls"] == 1

    def test_execute_failure(self):
        mgr = ErrorRecoveryManager()
        result = mgr.execute(
            lambda msgs, tokens, model: (_ for _ in ()).throw(Exception("unknown error")),
            [{"role": "user", "content": "hi"}],
        )
        assert result.success is False
        stats = mgr.stats()
        assert stats["failed_calls"] == 1

    def test_stats_tracking(self):
        mgr = ErrorRecoveryManager()

        def fn(msgs, tokens, model):
            return {"content": "ok"}

        mgr.execute(fn, [{"role": "user", "content": "hi"}])
        mgr.execute(fn, [{"role": "user", "content": "hi2"}])
        stats = mgr.stats()
        assert stats["total_calls"] == 2
        assert stats["successful_calls"] == 2

    def test_with_compact_fn(self):
        compacted = []

        def compact_fn(msgs):
            compacted.append(True)
            return [{"role": "user", "content": "compacted"}]

        mgr = ErrorRecoveryManager(compact_fn=compact_fn)
        attempts = [0]

        def fn(msgs, tokens, model):
            attempts[0] += 1
            if attempts[0] == 1:
                raise Exception("prompt_too_long")
            return {"content": "ok"}

        result = mgr.execute(fn, [{"role": "user", "content": "hi"}])
        assert result.success is True
        assert len(compacted) > 0
