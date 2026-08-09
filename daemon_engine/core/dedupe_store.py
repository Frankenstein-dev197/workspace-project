"""Dedupe store: idempotency for inbound messages and webhook redeliveries.

Integrates DeerFlow InboundDedupeStore pattern:
- DedupeStore: process-local OrderedDict store with TTL and max entries
- try_record: returns True if duplicate (drop), False if new (proceed)
- release: remove a key after processing
- TTL-based eviction: expired entries removed from front (chronological order)
- Max entries cap: prevents unbounded growth
- Thread-safe: all operations protected by locks

This guards agent runs and final answers against provider redeliveries.
A shared store (e.g., Postgres) can be injected for multi-pod deployments.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Protocol

logger = logging.getLogger(__name__)

DEFAULT_DEDUPE_TTL_SECONDS = 10 * 60  # 10 minutes
DEFAULT_DEDUPE_MAX_ENTRIES = 4096


class DedupeStore(Protocol):
    """Contract for recording/releasing dedupe keys.

    try_record returns True if the key already existed (duplicate -> drop)
    and False if newly recorded or prior entry had expired (proceed).
    """

    def try_record(self, key: tuple[Any, ...]) -> bool: ...
    def release(self, key: tuple[Any, ...]) -> None: ...
    def clear(self) -> None: ...
    def count(self) -> int: ...


class MemoryDedupeStore:
    """Process-local OrderedDict dedupe store.

    Preserves insertion order for chronological eviction of expired
    entries. Thread-safe for concurrent access.
    """

    def __init__(
        self,
        ttl_seconds: int = DEFAULT_DEDUPE_TTL_SECONDS,
        max_entries: int = DEFAULT_DEDUPE_MAX_ENTRIES,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_entries
        self._store: OrderedDict[tuple[Any, ...], float] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = {
            "total_recorded": 0,
            "total_duplicates": 0,
            "total_released": 0,
            "total_expired": 0,
        }

    def try_record(self, key: tuple[Any, ...]) -> bool:
        """Record a key. Returns True if duplicate, False if new."""
        with self._lock:
            now = time.monotonic()
            self._evict_expired(now)
            if key in self._store:
                self._stats["total_duplicates"] += 1
                return True
            self._store[key] = now
            self._stats["total_recorded"] += 1
            self._evict_overflow()
            return False

    def release(self, key: tuple[Any, ...]) -> None:
        """Remove a key after processing."""
        with self._lock:
            if key in self._store:
                self._store.pop(key)
                self._stats["total_released"] += 1

    def _evict_expired(self, now: float) -> int:
        """Evict expired entries from the front. Returns count evicted."""
        evicted = 0
        while self._store:
            _, oldest_at = next(iter(self._store.items()))
            if now - oldest_at > self._ttl:
                self._store.popitem(last=False)
                evicted += 1
            else:
                break
        self._stats["total_expired"] += evicted
        return evicted

    def _evict_overflow(self) -> int:
        """Evict oldest entries if over max. Returns count evicted."""
        evicted = 0
        while len(self._store) > self._max:
            self._store.popitem(last=False)
            evicted += 1
        return evicted

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._store.clear()

    def count(self) -> int:
        """Number of entries currently tracked."""
        with self._lock:
            return len(self._store)

    def contains(self, key: tuple[Any, ...]) -> bool:
        """Check if a key is tracked (non-destructive)."""
        with self._lock:
            return key in self._store

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "active_entries": len(self._store),
                "ttl_seconds": self._ttl,
                "max_entries": self._max,
            }


def make_dedupe_key(*parts: Any) -> tuple[Any, ...]:
    """Create a dedupe key from parts.

    Typical key: (channel_name, workspace_id, chat_id, message_id)
    """
    return tuple(parts)
