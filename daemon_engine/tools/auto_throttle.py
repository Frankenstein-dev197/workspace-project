"""Auto-throttle: adaptive per-domain rate limiting for web requests.

Integrates Scrapling AutoThrottle pattern:
- AutoThrottle: adjusts per-domain delay from observed response latency
  - Speeds up on fast servers, backs off on slow or hostile ones
  - start_delay: initial delay for first request to a domain
  - max_delay: highest delay allowed
  - target_concurrency: desired requests in flight per domain
  - block_backoff: double delay on block, or honor Retry-After header
- parse_retry_after: extract wait time from Retry-After header
  - Supports numeric seconds and HTTP date formats
- Per-domain delay tracking with thread-safe updates

This prevents overwhelming target servers and handles rate limiting
gracefully by adapting to each server's response characteristics.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

logger = logging.getLogger(__name__)


def parse_retry_after(headers: dict[str, str]) -> float | None:
    """Extract wait time from Retry-After header.

    Supports numeric seconds and HTTP date formats.
    Returns None when header is missing or unreadable.
    """
    value = next(
        (v for k, v in headers.items() if k.lower() == "retry-after"),
        "",
    ).strip()
    if not value:
        return None

    try:
        return max(float(value), 0.0)
    except ValueError:
        pass

    try:
        parsed = parsedate_to_datetime(value)
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        delta = (parsed - datetime.now(timezone.utc)).total_seconds()
        return max(delta, 0.0)
    except (TypeError, ValueError):
        logger.debug("Ignoring unreadable Retry-After header: %r", value)
        return None


@dataclass
class DomainStats:
    """Per-domain throttle statistics."""
    delay: float
    last_latency: float = 0.0
    total_requests: int = 0
    total_blocks: int = 0
    total_retries: int = 0


class AutoThrottle:
    """Adaptive per-domain rate limiting.

    Adjusts the per-domain delay from observed response latency, so the
    spider speeds up on fast servers and backs off on slow or hostile ones.
    """

    def __init__(
        self,
        start_delay: float = 5.0,
        max_delay: float = 60.0,
        target_concurrency: float = 1.0,
        block_backoff: bool = True,
    ) -> None:
        if target_concurrency <= 0:
            raise ValueError("target_concurrency must be higher than 0")
        if max_delay < start_delay:
            raise ValueError("max_delay can't be lower than start_delay")

        self.start_delay = start_delay
        self.max_delay = max_delay
        self.target_concurrency = target_concurrency
        self.block_backoff = block_backoff
        self._delays: dict[str, float] = {}
        self._stats: dict[str, DomainStats] = {}
        self._lock = threading.Lock()

    def get_delay(self, domain: str) -> float:
        """Get current delay for a domain (defaults to start_delay)."""
        with self._lock:
            return self._delays.get(domain, self.start_delay)

    def set_delay(self, domain: str, delay: float) -> None:
        """Set delay for a domain, clamped to [0, max_delay]."""
        with self._lock:
            clamped = min(max(delay, 0.0), self.max_delay)
            self._delays[domain] = clamped
            stats = self._stats.setdefault(domain, DomainStats(delay=clamped))
            stats.delay = clamped

    def record_request(
        self,
        domain: str,
        latency: float,
        blocked: bool = False,
        retry_after: float | None = None,
    ) -> float:
        """Record a request and adjust the delay.

        Returns the new delay for the domain.
        """
        with self._lock:
            current = self._delays.get(domain, self.start_delay)
            stats = self._stats.setdefault(domain, DomainStats(delay=current))
            stats.total_requests += 1
            stats.last_latency = latency

            if blocked:
                stats.total_blocks += 1
                if retry_after is not None:
                    new_delay = max(retry_after, current)
                elif self.block_backoff:
                    new_delay = min(current * 2, self.max_delay)
                else:
                    new_delay = current
            else:
                # Adjust based on latency: fast response → reduce delay
                if latency < current * 0.5:
                    new_delay = max(current * 0.75, self.start_delay * 0.5)
                elif latency > current * 2:
                    new_delay = min(current * 1.5, self.max_delay)
                else:
                    new_delay = current

            clamped = min(max(new_delay, 0.0), self.max_delay)
            self._delays[domain] = clamped
            stats.delay = clamped
            return clamped

    def record_retry(self, domain: str) -> None:
        """Record a retry attempt for a domain."""
        with self._lock:
            stats = self._stats.setdefault(
                domain, DomainStats(delay=self._delays.get(domain, self.start_delay))
            )
            stats.total_retries += 1

    def reset_domain(self, domain: str) -> None:
        """Reset a domain's delay to start_delay."""
        with self._lock:
            self._delays.pop(domain, None)
            self._stats.pop(domain, None)

    def clear(self) -> None:
        """Clear all domain delays and stats."""
        with self._lock:
            self._delays.clear()
            self._stats.clear()

    def get_stats(self, domain: str) -> DomainStats | None:
        """Get stats for a domain."""
        with self._lock:
            return self._stats.get(domain)

    def all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all domains."""
        with self._lock:
            return {
                domain: {
                    "delay": s.delay,
                    "last_latency": s.last_latency,
                    "total_requests": s.total_requests,
                    "total_blocks": s.total_blocks,
                    "total_retries": s.total_retries,
                }
                for domain, s in self._stats.items()
            }

    def domain_count(self) -> int:
        """Number of tracked domains."""
        with self._lock:
            return len(self._delays)
