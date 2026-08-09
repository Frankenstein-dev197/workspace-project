"""Error recovery: LLM call resilience with backoff and fallback.

Integrates learn-claude-code's s11_error_recovery pattern: three recovery
paths for LLM API errors with exponential backoff, token escalation,
prompt compaction, and model fallback.

Recovery paths:
1. max_tokens → escalate token limit, then continuation prompt
2. prompt_too_long → reactive compact → retry
3. 429/529 rate limit → exponential backoff with jitter, fallback model
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOKENS = 8000
ESCALATED_MAX_TOKENS = 64000
MAX_RECOVERY_RETRIES = 3
MAX_RETRIES = 10
BASE_DELAY_MS = 500
MAX_CONSECUTIVE_529 = 3
MAX_CONTINUATION_RETRIES = 3

CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — "
    "no apology, no recap. Pick up mid-thought."
)


class ErrorType(str, Enum):
    """Types of LLM API errors."""
    MAX_TOKENS = "max_tokens"
    PROMPT_TOO_LONG = "prompt_too_long"
    RATE_LIMIT = "rate_limit"
    OVERLOADED = "overloaded"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    UNKNOWN = "unknown"


class RecoveryAction(str, Enum):
    """Recovery actions for errors."""
    ESCALATE_TOKENS = "escalate_tokens"
    CONTINUATION = "continuation"
    COMPACT = "compact"
    BACKOFF = "backoff"
    FALLBACK_MODEL = "fallback_model"
    RETRY = "retry"
    GIVE_UP = "give_up"


@dataclass
class RecoveryState:
    """Tracks recovery state across retries."""
    token_escalated: bool = False
    compacted: bool = False
    consecutive_529: int = 0
    model_switched: bool = False
    continuation_count: int = 0
    total_retries: int = 0
    current_max_tokens: int = DEFAULT_MAX_TOKENS
    current_model: str = ""


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    response: Any = None
    error: str = ""
    error_type: ErrorType | None = None
    action_taken: RecoveryAction | None = None
    retries: int = 0
    duration: float = 0.0


def classify_error(exc: Exception) -> ErrorType:
    """Classify an exception into an error type."""
    exc_str = str(exc).lower()
    exc_type = type(exc).__name__.lower()
    if "max_tokens" in exc_str or "max output tokens" in exc_str:
        return ErrorType.MAX_TOKENS
    if "prompt_too_long" in exc_str or "context length" in exc_str or "too long" in exc_str:
        return ErrorType.PROMPT_TOO_LONG
    if "429" in exc_str or "rate_limit" in exc_type or "rate limit" in exc_str:
        return ErrorType.RATE_LIMIT
    if "529" in exc_str or "overloaded" in exc_str or "overloaded_error" in exc_type:
        return ErrorType.OVERLOADED
    if "timeout" in exc_str or "timed out" in exc_str:
        return ErrorType.TIMEOUT
    if "connection" in exc_str or "connectionerror" in exc_type:
        return ErrorType.CONNECTION
    return ErrorType.UNKNOWN


def compute_backoff_delay(
    attempt: int,
    base_delay_ms: int = BASE_DELAY_MS,
    max_delay_ms: int = 30000,
) -> float:
    """Compute exponential backoff delay with jitter."""
    delay = min(base_delay_ms * (2 ** attempt), max_delay_ms)
    jitter = random.uniform(0, delay * 0.1)
    return (delay + jitter) / 1000.0


def determine_action(
    error_type: ErrorType,
    state: RecoveryState,
) -> RecoveryAction:
    """Determine the recovery action for an error."""
    if error_type == ErrorType.MAX_TOKENS:
        if not state.token_escalated:
            return RecoveryAction.ESCALATE_TOKENS
        if state.continuation_count < MAX_CONTINUATION_RETRIES:
            return RecoveryAction.CONTINUATION
        return RecoveryAction.GIVE_UP
    if error_type == ErrorType.PROMPT_TOO_LONG:
        if not state.compacted:
            return RecoveryAction.COMPACT
        return RecoveryAction.GIVE_UP
    if error_type in (ErrorType.RATE_LIMIT, ErrorType.OVERLOADED):
        if error_type == ErrorType.OVERLOADED:
            state.consecutive_529 += 1
            if state.consecutive_529 >= MAX_CONSECUTIVE_529 and not state.model_switched:
                return RecoveryAction.FALLBACK_MODEL
        return RecoveryAction.BACKOFF
    if error_type in (ErrorType.TIMEOUT, ErrorType.CONNECTION):
        if state.total_retries < MAX_RETRIES:
            return RecoveryAction.RETRY
        return RecoveryAction.GIVE_UP
    return RecoveryAction.GIVE_UP


def apply_action(
    action: RecoveryAction,
    state: RecoveryState,
    messages: list[dict[str, Any]] | None = None,
    fallback_model: str | None = None,
    compact_fn: Callable | None = None,
) -> tuple[list[dict[str, Any]], int, str]:
    """Apply a recovery action. Returns (messages, max_tokens, model)."""
    msgs = list(messages or [])
    max_tokens = state.current_max_tokens
    model = state.current_model

    if action == RecoveryAction.ESCALATE_TOKENS:
        state.token_escalated = True
        state.current_max_tokens = ESCALATED_MAX_TOKENS
        max_tokens = ESCALATED_MAX_TOKENS
        logger.info("Escalating max_tokens to %d", max_tokens)

    elif action == RecoveryAction.CONTINUATION:
        state.continuation_count += 1
        msgs.append({"role": "user", "content": CONTINUATION_PROMPT})
        logger.info("Adding continuation prompt (attempt %d)", state.continuation_count)

    elif action == RecoveryAction.COMPACT:
        state.compacted = True
        if compact_fn:
            msgs = compact_fn(msgs)
        logger.info("Applying reactive compaction")

    elif action == RecoveryAction.BACKOFF:
        delay = compute_backoff_delay(state.total_retries)
        logger.info("Backing off for %.2fs (attempt %d)", delay, state.total_retries)
        time.sleep(delay)

    elif action == RecoveryAction.FALLBACK_MODEL:
        state.model_switched = True
        if fallback_model:
            state.current_model = fallback_model
            model = fallback_model
            logger.info("Switching to fallback model: %s", model)

    return msgs, max_tokens, model


def with_retry(
    fn: Callable,
    messages: list[dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    model: str = "",
    fallback_model: str | None = None,
    compact_fn: Callable | None = None,
    max_retries: int = MAX_RETRIES,
) -> RecoveryResult:
    """Execute a function with retry and recovery logic.

    Args:
        fn: Callable(messages, max_tokens, model) -> response
        messages: Initial message list
        max_tokens: Initial max token limit
        model: Initial model name
        fallback_model: Fallback model for consecutive 529s
        compact_fn: Function(messages) -> compacted messages
        max_retries: Maximum retry attempts

    Returns:
        RecoveryResult with response or error
    """
    start_time = time.time()
    state = RecoveryState(
        current_max_tokens=max_tokens,
        current_model=model,
    )
    current_messages = list(messages)

    while state.total_retries < max_retries:
        try:
            response = fn(current_messages, state.current_max_tokens, state.current_model)
            return RecoveryResult(
                success=True,
                response=response,
                retries=state.total_retries,
                duration=time.time() - start_time,
            )
        except Exception as exc:
            state.total_retries += 1
            error_type = classify_error(exc)
            action = determine_action(error_type, state)

            logger.warning(
                "Error %s on attempt %d, action: %s",
                error_type.value, state.total_retries, action.value,
            )

            if action == RecoveryAction.GIVE_UP:
                return RecoveryResult(
                    success=False,
                    error=str(exc),
                    error_type=error_type,
                    action_taken=action,
                    retries=state.total_retries,
                    duration=time.time() - start_time,
                )

            current_messages, state.current_max_tokens, state.current_model = apply_action(
                action,
                state,
                current_messages,
                fallback_model,
                compact_fn,
            )

    return RecoveryResult(
        success=False,
        error=f"Exceeded max retries ({max_retries})",
        error_type=ErrorType.UNKNOWN,
        action_taken=RecoveryAction.GIVE_UP,
        retries=state.total_retries,
        duration=time.time() - start_time,
    )


class ErrorRecoveryManager:
    """Manages error recovery for LLM calls.

    Wraps LLM calls with retry logic, token escalation, prompt compaction,
    and model fallback. Tracks recovery statistics.
    """

    def __init__(
        self,
        fallback_model: str | None = None,
        compact_fn: Callable | None = None,
    ) -> None:
        self.fallback_model = fallback_model
        self.compact_fn = compact_fn
        self._stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "retries": 0,
            "token_escalations": 0,
            "compactions": 0,
            "model_fallbacks": 0,
            "backoffs": 0,
            "by_error_type": {et.value: 0 for et in ErrorType},
        }

    def execute(
        self,
        fn: Callable,
        messages: list[dict[str, Any]],
        max_tokens: int = DEFAULT_MAX_TOKENS,
        model: str = "",
    ) -> RecoveryResult:
        """Execute an LLM call with error recovery."""
        self._stats["total_calls"] += 1
        result = with_retry(
            fn,
            messages,
            max_tokens=max_tokens,
            model=model,
            fallback_model=self.fallback_model,
            compact_fn=self.compact_fn,
        )
        if result.success:
            self._stats["successful_calls"] += 1
        else:
            self._stats["failed_calls"] += 1
        self._stats["retries"] += result.retries
        if result.error_type:
            self._stats["by_error_type"][result.error_type.value] += 1
        if result.action_taken == RecoveryAction.ESCALATE_TOKENS:
            self._stats["token_escalations"] += 1
        elif result.action_taken == RecoveryAction.COMPACT:
            self._stats["compactions"] += 1
        elif result.action_taken == RecoveryAction.FALLBACK_MODEL:
            self._stats["model_fallbacks"] += 1
        elif result.action_taken == RecoveryAction.BACKOFF:
            self._stats["backoffs"] += 1
        return result

    def stats(self) -> dict[str, Any]:
        return dict(self._stats)
