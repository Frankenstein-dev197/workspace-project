"""LLM cache: prompt caching with drift detection.

Integrates Headroom's cache stabilization pattern: structural hashing
of the cache hot zone (system prompt, tools, early messages) to detect
cache-busting drift and enable prompt caching for LLM calls.

When the system prompt and tools are stable across calls, the LLM
provider can cache the prefix and charge less for subsequent calls.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

EARLY_MESSAGES_WINDOW = 3


class ApiKind(str, Enum):
    """Provider body shape (from Headroom ApiKind)."""
    ANTHROPIC = "anthropic"
    OPENAI_CHAT = "openai_chat"
    OPENAI_RESPONSES = "openai_responses"
    GENERIC = "generic"


@dataclass
class StructuralHash:
    """Three-axis fingerprint of the cache hot zone."""
    system: str = ""
    tools: str = ""
    early_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "tools": self.tools,
            "early_messages": self.early_messages,
        }


@dataclass
class CacheEntry:
    """A cached LLM response."""
    key: str
    response: Any
    created_at: float = field(default_factory=time.time)
    hit_count: int = 0
    token_count: int = 0
    api_kind: ApiKind = ApiKind.GENERIC


@dataclass
class DriftReport:
    """Report of cache drift between two requests."""
    session_id: str
    drifted_dimensions: list[str] = field(default_factory=list)
    is_drift: bool = False
    is_first_request: bool = False
    previous_hash: StructuralHash | None = None
    current_hash: StructuralHash | None = None
    timestamp: float = field(default_factory=time.time)


def canonicalize_for_hash(obj: Any) -> Any:
    """Canonicalize an object for stable hashing (sorted keys, no whitespace)."""
    if isinstance(obj, dict):
        return {k: canonicalize_for_hash(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [canonicalize_for_hash(item) for item in obj]
    return obj


def sha256_hex(data: Any) -> str:
    """Compute SHA-256 hex of canonical JSON."""
    canonical = canonicalize_for_hash(data)
    encoded = json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def extract_cache_hot_zone(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    system_prompt: str = "",
    api_kind: ApiKind = ApiKind.GENERIC,
) -> StructuralHash:
    """Extract the structural hash of the cache hot zone."""
    system_hash = sha256_hex(system_prompt) if system_prompt else ""
    tools_hash = sha256_hex(tools) if tools else ""
    early = messages[:EARLY_MESSAGES_WINDOW]
    early_hashes = [sha256_hex(msg) for msg in early]
    return StructuralHash(
        system=system_hash,
        tools=tools_hash,
        early_messages=early_hashes,
    )


def compute_drift(
    previous: StructuralHash,
    current: StructuralHash,
) -> list[str]:
    """Compute which dimensions drifted between two hashes."""
    drifted: list[str] = []
    if previous.system != current.system:
        drifted.append("system")
    if previous.tools != current.tools:
        drifted.append("tools")
    max_len = max(len(previous.early_messages), len(current.early_messages))
    for i in range(max_len):
        prev_hash = previous.early_messages[i] if i < len(previous.early_messages) else None
        curr_hash = current.early_messages[i] if i < len(current.early_messages) else None
        if prev_hash is not None and curr_hash is not None and prev_hash != curr_hash:
            drifted.append(f"early_message[{i}]")
    return drifted


def conversation_discriminator(messages: list[dict[str, Any]]) -> str:
    """Generate a session key from the first message (privacy-preserving)."""
    if not messages:
        return "empty"
    first = messages[0]
    content = str(first.get("content", ""))[:100]
    return sha256_hex(content)[:16]


class LLMCache:
    """Prompt cache with drift detection.

    Caches LLM responses keyed by structural hash of the prompt.
    Detects when the cache hot zone drifts (system prompt, tools, or
    early messages change) and logs the drift dimensions.
    """

    def __init__(self, max_entries: int = 1000, ttl_seconds: int = 3600) -> None:
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._session_hashes: dict[str, StructuralHash] = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "drifts_detected": 0,
            "first_requests": 0,
        }

    def _make_key(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        model: str = "",
        api_kind: ApiKind = ApiKind.GENERIC,
    ) -> str:
        hot_zone = extract_cache_hot_zone(messages, tools, system_prompt, api_kind)
        last_message = messages[-1] if messages else {}
        last_hash = sha256_hex(last_message)
        key_parts = [
            model,
            hot_zone.system,
            hot_zone.tools,
            last_hash,
        ]
        return sha256_hex("|".join(key_parts))

    def get(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        model: str = "",
        api_kind: ApiKind = ApiKind.GENERIC,
    ) -> tuple[Any, DriftReport | None]:
        """Get a cached response. Returns (response_or_None, drift_report)."""
        session_id = conversation_discriminator(messages)
        current_hash = extract_cache_hot_zone(messages, tools, system_prompt, api_kind)
        previous_hash = self._session_hashes.get(session_id)
        drift_report: DriftReport | None = None
        if previous_hash is None:
            drift_report = DriftReport(
                session_id=session_id,
                is_first_request=True,
                current_hash=current_hash,
            )
            self._stats["first_requests"] += 1
            logger.debug("Cache first request for session %s", session_id)
        else:
            drifted = compute_drift(previous_hash, current_hash)
            if drifted:
                drift_report = DriftReport(
                    session_id=session_id,
                    drifted_dimensions=drifted,
                    is_drift=True,
                    previous_hash=previous_hash,
                    current_hash=current_hash,
                )
                self._stats["drifts_detected"] += 1
                logger.info(
                    "Cache drift for session %s: %s",
                    session_id, ", ".join(drifted),
                )
        self._session_hashes[session_id] = current_hash
        key = self._make_key(messages, tools, system_prompt, model, api_kind)
        entry = self._cache.get(key)
        if entry is None:
            self._stats["misses"] += 1
            return None, drift_report
        if time.time() - entry.created_at > self._ttl_seconds:
            del self._cache[key]
            self._stats["misses"] += 1
            return None, drift_report
        entry.hit_count += 1
        self._stats["hits"] += 1
        self._cache.move_to_end(key)
        return entry.response, drift_report

    def put(
        self,
        messages: list[dict[str, Any]],
        response: Any,
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        model: str = "",
        api_kind: ApiKind = ApiKind.GENERIC,
        token_count: int = 0,
    ) -> str:
        """Store a response in the cache. Returns the cache key."""
        key = self._make_key(messages, tools, system_prompt, model, api_kind)
        entry = CacheEntry(
            key=key,
            response=response,
            api_kind=api_kind,
            token_count=token_count,
        )
        self._cache[key] = entry
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_entries:
            evicted_key, _ = self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            logger.debug("Evicted cache entry %s", evicted_key)
        return key

    def invalidate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
        model: str = "",
        api_kind: ApiKind = ApiKind.GENERIC,
    ) -> bool:
        """Remove a specific entry from the cache."""
        key = self._make_key(messages, tools, system_prompt, model, api_kind)
        return self._cache.pop(key, None) is not None

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._session_hashes.clear()

    def stats(self) -> dict[str, Any]:
        hit_rate = 0.0
        total = self._stats["hits"] + self._stats["misses"]
        if total > 0:
            hit_rate = self._stats["hits"] / total
        return {
            **self._stats,
            "hit_rate": round(hit_rate, 4),
            "cache_size": len(self._cache),
            "max_entries": self._max_entries,
            "ttl_seconds": self._ttl_seconds,
            "tracked_sessions": len(self._session_hashes),
        }

    def get_drift_history(self) -> dict[str, StructuralHash]:
        """Get the session hash history for drift analysis."""
        return dict(self._session_hashes)

    def check_drift(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str = "",
    ) -> DriftReport:
        """Check for drift without getting/putting from cache."""
        session_id = conversation_discriminator(messages)
        current_hash = extract_cache_hot_zone(messages, tools, system_prompt)
        previous_hash = self._session_hashes.get(session_id)
        if previous_hash is None:
            report = DriftReport(
                session_id=session_id,
                is_first_request=True,
                current_hash=current_hash,
            )
        else:
            drifted = compute_drift(previous_hash, current_hash)
            report = DriftReport(
                session_id=session_id,
                drifted_dimensions=drifted,
                is_drift=bool(drifted),
                previous_hash=previous_hash,
                current_hash=current_hash,
            )
        self._session_hashes[session_id] = current_hash
        return report
