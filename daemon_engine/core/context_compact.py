"""Context compaction: multi-layer message history management.

Integrates learn-claude-code's s08_context_compact 4-layer pipeline:

L1: snip_compact - trim middle messages when count exceeds threshold
L2: micro_compact - replace old tool results with placeholders
L3: tool_result_budget - persist large outputs to disk, keep preview
L4: compact_history - LLM full summary (expensive, last resort)

Emergency: reactive_compact - on prompt_too_long API error

Core principle: cheap first, expensive last.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MAX_MESSAGES_THRESHOLD = 50
KEEP_RECENT_TOOL_RESULTS = 6
PERSIST_THRESHOLD = 5000
MAX_TOOL_RESULT_BYTES = 200_000
SUMMARY_MAX_CHARS = 80000


@dataclass
class CompactionStats:
    """Statistics for compaction operations."""
    snip_compactions: int = 0
    micro_compactions: int = 0
    budget_persists: int = 0
    full_summaries: int = 0
    reactive_compactions: int = 0
    messages_snipped: int = 0
    tool_results_compacted: int = 0
    bytes_persisted: int = 0


def _is_tool_result_message(msg: dict[str, Any]) -> bool:
    """Check if a message contains tool results."""
    content = msg.get("content")
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_result"
            for b in content
        )
    return False


def _message_has_tool_use(msg: dict[str, Any]) -> bool:
    """Check if a message contains tool_use blocks."""
    content = msg.get("content")
    if isinstance(content, list):
        return any(
            isinstance(b, dict) and b.get("type") == "tool_use"
            for b in content
        )
    return False


def _collect_tool_results(messages: list[dict[str, Any]]) -> list[tuple[int, int, dict[str, Any]]]:
    """Collect all tool_result blocks with their positions."""
    blocks: list[tuple[int, int, dict[str, Any]]] = []
    for mi, msg in enumerate(messages):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        for bi, block in enumerate(msg["content"]):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                blocks.append((mi, bi, block))
    return blocks


def snip_compact(
    messages: list[dict[str, Any]],
    max_messages: int = MAX_MESSAGES_THRESHOLD,
) -> list[dict[str, Any]]:
    """L1: Trim middle messages when count exceeds threshold.

    Keeps head (first 3) and tail (max_messages - 3) messages.
    Handles tool_use/tool_result pairs to avoid orphaning.
    """
    if len(messages) <= max_messages:
        return messages
    keep_head = 3
    keep_tail = max_messages - keep_head
    head_end = keep_head
    tail_start = len(messages) - keep_tail
    if head_end > 0 and _message_has_tool_use(messages[head_end - 1]):
        while head_end < len(messages) and _is_tool_result_message(messages[head_end]):
            head_end += 1
    if (
        tail_start > 0
        and tail_start < len(messages)
        and _is_tool_result_message(messages[tail_start])
        and _message_has_tool_use(messages[tail_start - 1])
    ):
        tail_start -= 1
    if head_end >= tail_start:
        return messages
    snipped = tail_start - head_end
    logger.debug("snip_compact: removed %d messages", snipped)
    return (
        messages[:head_end]
        + [{"role": "user", "content": f"[snipped {snipped} messages]"}]
        + messages[tail_start:]
    )


def micro_compact(
    messages: list[dict[str, Any]],
    keep_recent: int = KEEP_RECENT_TOOL_RESULTS,
) -> list[dict[str, Any]]:
    """L2: Replace old tool results with placeholders."""
    tool_results = _collect_tool_results(messages)
    if len(tool_results) <= keep_recent:
        return messages
    compacted = 0
    for _, _, block in tool_results[:-keep_recent]:
        content = block.get("content", "")
        if len(str(content)) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"
            compacted += 1
    if compacted:
        logger.debug("micro_compact: replaced %d tool results", compacted)
    return messages


def persist_large_output(
    tool_use_id: str,
    output: str,
    persist_dir: Path,
) -> str:
    """L3: Persist large tool output to disk, return preview."""
    if len(output) <= PERSIST_THRESHOLD:
        return output
    persist_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in tool_use_id)
    path = persist_dir / f"{safe_id}.txt"
    if not path.exists():
        path.write_text(output)
        logger.debug("Persisted large output to %s (%d bytes)", path, len(output))
    return (
        f"<persisted-output>\n"
        f"Full output: {path}\n"
        f"Preview:\n{output[:2000]}\n"
        f"</persisted-output>"
    )


def tool_result_budget(
    messages: list[dict[str, Any]],
    max_bytes: int = MAX_TOOL_RESULT_BYTES,
    persist_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """L3: Enforce byte budget on tool results by persisting large ones."""
    if persist_dir is None:
        persist_dir = Path(".task_outputs") / "tool-results"
    last = messages[-1] if messages else None
    if (
        not last
        or last.get("role") != "user"
        or not isinstance(last.get("content"), list)
    ):
        return messages
    blocks = [
        (i, b) for i, b in enumerate(last["content"])
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    if total <= max_bytes:
        return messages
    ranked = sorted(blocks, key=lambda p: len(str(p[1].get("content", ""))), reverse=True)
    for _, block in ranked:
        if total <= max_bytes:
            break
        content = str(block.get("content", ""))
        if len(content) <= PERSIST_THRESHOLD:
            continue
        tid = block.get("tool_use_id", "unknown")
        persisted = persist_large_output(tid, content, persist_dir)
        block["content"] = persisted
        total = sum(len(str(b.get("content", ""))) for _, b in blocks)
    return messages


def write_transcript(
    messages: list[dict[str, Any]],
    transcript_dir: Path,
) -> Path:
    """Write message history to a transcript file for recovery."""
    transcript_dir.mkdir(parents=True, exist_ok=True)
    path = transcript_dir / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    return path


def summarize_history_static(
    messages: list[dict[str, Any]],
    max_chars: int = SUMMARY_MAX_CHARS,
) -> str:
    """Static summary without LLM call (for testing/fallback)."""
    parts: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        texts.append(block.get("text", "")[:200])
                    elif block.get("type") == "tool_use":
                        texts.append(f"[tool: {block.get('name', '')}]")
                    elif block.get("type") == "tool_result":
                        texts.append("[tool_result]")
            content = " ".join(texts)
        content_str = str(content)[:200]
        parts.append(f"[{role}] {content_str}")
    summary = "\n".join(parts)
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "\n... [truncated]"
    return summary


def compact_history(
    messages: list[dict[str, Any]],
    transcript_dir: Path | None = None,
    summarize_fn: Any = None,
) -> list[dict[str, Any]]:
    """L4: Full history compaction with LLM summary.

    Args:
        messages: The message history to compact.
        transcript_dir: Directory to save transcript before compaction.
        summarize_fn: Function(messages) -> summary string.
                      If None, uses static summary.
    """
    if transcript_dir:
        write_transcript(messages, transcript_dir)
    if summarize_fn:
        summary = summarize_fn(messages)
    else:
        summary = summarize_history_static(messages)
    logger.info("compact_history: summarized %d messages", len(messages))
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]


def reactive_compact(
    messages: list[dict[str, Any]],
    transcript_dir: Path | None = None,
    summarize_fn: Any = None,
    keep_tail: int = 5,
) -> list[dict[str, Any]]:
    """Emergency compaction on prompt_too_long error.

    Keeps the last few messages and summarizes the rest.
    """
    if transcript_dir:
        write_transcript(messages, transcript_dir)
    tail_start = max(0, len(messages) - keep_tail)
    if (
        tail_start > 0
        and tail_start < len(messages)
        and _is_tool_result_message(messages[tail_start])
        and _message_has_tool_use(messages[tail_start - 1])
    ):
        tail_start -= 1
    to_summarize = messages[:tail_start]
    if summarize_fn:
        summary = summarize_fn(to_summarize)
    else:
        summary = summarize_history_static(to_summarize)
    logger.warning("reactive_compact: emergency compaction of %d messages", len(to_summarize))
    return (
        [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}]
        + messages[tail_start:]
    )


def estimate_tokens(messages: list[dict[str, Any]], chars_per_token: float = 4.0) -> int:
    """Estimate token count for messages."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    total_chars += len(str(block.get("text", block.get("content", ""))))
        else:
            total_chars += len(str(content))
    return int(total_chars / chars_per_token)


def needs_compaction(
    messages: list[dict[str, Any]],
    max_messages: int = MAX_MESSAGES_THRESHOLD,
    max_tokens: int = 100000,
) -> str | None:
    """Check if compaction is needed. Returns compaction level or None."""
    if len(messages) > max_messages:
        return "snip"
    tokens = estimate_tokens(messages)
    if tokens > max_tokens:
        return "full"
    return None


def auto_compact(
    messages: list[dict[str, Any]],
    max_messages: int = MAX_MESSAGES_THRESHOLD,
    max_tokens: int = 100000,
    persist_dir: Path | None = None,
    transcript_dir: Path | None = None,
    summarize_fn: Any = None,
) -> tuple[list[dict[str, Any]], str]:
    """Run the full compaction pipeline automatically.

    Returns (compacted_messages, compaction_level_used).
    """
    result = list(messages)
    level = "none"
    result = tool_result_budget(result, persist_dir=persist_dir)
    result = snip_compact(result, max_messages=max_messages)
    if len(result) < len(messages):
        level = "snip"
    result = micro_compact(result)
    tokens = estimate_tokens(result)
    if tokens > max_tokens:
        result = compact_history(
            result,
            transcript_dir=transcript_dir,
            summarize_fn=summarize_fn,
        )
        level = "full"
    return result, level


class ContextCompactor:
    """Context compaction manager with stats tracking.

    Provides the 4-layer compaction pipeline with automatic triggering
    based on message count and token estimates.
    """

    def __init__(
        self,
        persist_dir: Path | None = None,
        transcript_dir: Path | None = None,
        max_messages: int = MAX_MESSAGES_THRESHOLD,
        max_tokens: int = 100000,
    ) -> None:
        self.persist_dir = persist_dir or Path(".task_outputs") / "tool-results"
        self.transcript_dir = transcript_dir or Path(".transcripts")
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.stats = CompactionStats()

    def compact(
        self,
        messages: list[dict[str, Any]],
        summarize_fn: Any = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Run auto-compaction pipeline."""
        result, level = auto_compact(
            messages,
            max_messages=self.max_messages,
            max_tokens=self.max_tokens,
            persist_dir=self.persist_dir,
            transcript_dir=self.transcript_dir,
            summarize_fn=summarize_fn,
        )
        if level == "snip":
            self.stats.snip_compactions += 1
            self.stats.messages_snipped += len(messages) - len(result)
        elif level == "full":
            self.stats.full_summaries += 1
        return result, level

    def reactive(
        self,
        messages: list[dict[str, Any]],
        summarize_fn: Any = None,
    ) -> list[dict[str, Any]]:
        """Emergency reactive compaction."""
        self.stats.reactive_compactions += 1
        return reactive_compact(
            messages,
            transcript_dir=self.transcript_dir,
            summarize_fn=summarize_fn,
        )

    def check_needed(self, messages: list[dict[str, Any]]) -> str | None:
        """Check if compaction is needed."""
        return needs_compaction(messages, self.max_messages, self.max_tokens)

    def stats_dict(self) -> dict[str, Any]:
        return {
            "snip_compactions": self.stats.snip_compactions,
            "micro_compactions": self.stats.micro_compactions,
            "budget_persists": self.stats.budget_persists,
            "full_summaries": self.stats.full_summaries,
            "reactive_compactions": self.stats.reactive_compactions,
            "messages_snipped": self.stats.messages_snipped,
            "tool_results_compacted": self.stats.tool_results_compacted,
            "bytes_persisted": self.stats.bytes_persisted,
            "max_messages": self.max_messages,
            "max_tokens": self.max_tokens,
        }
