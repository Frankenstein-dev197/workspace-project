"""Priority scheduler: URL/request priority queue with deduplication.

Integrates Scrapling Scheduler pattern:
- PriorityScheduler: thread-safe priority queue with URL deduplication
  - heapq-based: higher priority requests dequeued first
  - URL fingerprinting: canonical URL normalization for dedup
  - dont_filter: allow duplicates when explicitly requested
  - In-flight tracking: requests tracked until complete()
  - Snapshot/restore: checkpoint state for resumption
- Request fingerprinting: canonicalize URL, include method/body
- Snapshot: capture pending + seen for checkpointing
- Restore: rebuild scheduler state from checkpoint

Useful for crawl/scrape orchestration where requests must be ordered
by priority and duplicate URLs must be filtered.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

logger = logging.getLogger(__name__)


def canonicalize_url(url: str, keep_fragments: bool = False) -> str:
    """Normalize a URL for deduplication.

    - Lowercase scheme and host
    - Remove default port
    - Sort query parameters
    - Optionally strip fragment
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    # Remove default port
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[:-3]
    elif netloc.endswith(":443") and scheme == "https":
        netloc = netloc[:-4]

    # Sort query params
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))

    fragment = parsed.fragment if keep_fragments else ""
    return urlunparse((scheme, netloc, parsed.path, parsed.params, query, fragment))


def fingerprint_url(
    url: str,
    method: str = "GET",
    body: bytes = b"",
    keep_fragments: bool = False,
) -> bytes:
    """Generate a unique fingerprint for a URL request.

    Includes canonical URL, method, and body hash for deduplication.
    """
    canon = canonicalize_url(url, keep_fragments=keep_fragments)
    data = f"{method.upper()}|{canon}|{body.hex()}"
    return hashlib.sha256(data.encode()).digest()


@dataclass(order=True)
class ScheduledRequest:
    """A request in the priority queue.

    sorted by (priority, counter) where lower priority value = higher
    actual priority (heapq is a min-heap).
    """
    sort_key: tuple[int, int]
    request_id: int = field(compare=False)
    url: str = field(compare=False)
    priority: int = field(compare=False, default=0)
    method: str = field(compare=False, default="GET")
    body: bytes = field(compare=False, default=b"")
    meta: dict[str, Any] = field(compare=False, default_factory=dict)
    dont_filter: bool = field(compare=False, default=False)

    def fingerprint(self, keep_fragments: bool = False) -> bytes:
        return fingerprint_url(
            self.url, self.method, self.body, keep_fragments
        )


@dataclass
class CheckpointData:
    """Snapshot of scheduler state for checkpointing."""
    pending: list[ScheduledRequest]
    seen: set[bytes]


class PriorityScheduler:
    """Thread-safe priority queue with URL deduplication.

    Higher priority requests are processed first. Duplicate URLs are
    filtered unless dont_filter=True. Requests are tracked as in-flight
    until complete() is called.
    """

    def __init__(self, keep_fragments: bool = False) -> None:
        self._heap: list[ScheduledRequest] = []
        self._seen: set[bytes] = set()
        self._counter = Counter()
        self._next_id = 0
        self._pending: dict[int, ScheduledRequest] = {}
        self._inflight: dict[int, list[int]] = {}
        self._keep_fragments = keep_fragments
        self._lock = threading.Lock()

    def enqueue(
        self,
        url: str,
        priority: int = 0,
        method: str = "GET",
        body: bytes = b"",
        meta: dict[str, Any] | None = None,
        dont_filter: bool = False,
    ) -> bool:
        """Add a request to the queue.

        Returns True if added, False if dropped as duplicate.
        """
        import heapq

        with self._lock:
            temp_req = ScheduledRequest(
                sort_key=(0, 0),
                request_id=0,
                url=url,
                priority=priority,
                method=method,
                body=body,
                meta=meta or {},
                dont_filter=dont_filter,
            )
            fp = temp_req.fingerprint(self._keep_fragments)

            if not dont_filter and fp in self._seen:
                logger.debug("Dropped duplicate request: %s", url)
                return False

            self._seen.add(fp)
            self._next_id += 1
            req_id = self._next_id
            # Negative priority so higher priority = dequeued first (min-heap)
            sort_key = (-priority, req_id)
            req = ScheduledRequest(
                sort_key=sort_key,
                request_id=req_id,
                url=url,
                priority=priority,
                method=method,
                body=body,
                meta=meta or {},
                dont_filter=dont_filter,
            )
            heapq.heappush(self._heap, req)
            self._pending[req_id] = req
            return True

    def dequeue(self) -> ScheduledRequest | None:
        """Get the next request to process (stays tracked until complete())."""
        import heapq

        with self._lock:
            if not self._heap:
                return None
            req = heapq.heappop(self._heap)
            self._pending.pop(req.request_id, None)
            self._inflight.setdefault(req.request_id, [])
            return req

    def complete(self, req_id: int) -> None:
        """Mark a request as finished."""
        with self._lock:
            self._inflight.pop(req_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._heap) == 0

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._heap)

    @property
    def inflight_count(self) -> int:
        with self._lock:
            return len(self._inflight)

    @property
    def seen_count(self) -> int:
        with self._lock:
            return len(self._seen)

    def snapshot(self) -> CheckpointData:
        """Create a snapshot of current state for checkpoints."""
        with self._lock:
            sorted_reqs = sorted(self._pending.values(), key=lambda r: r.sort_key)
            return CheckpointData(
                pending=list(sorted_reqs),
                seen=self._seen.copy(),
            )

    def restore(self, data: CheckpointData) -> None:
        """Restore scheduler state from checkpoint data."""
        import heapq

        with self._lock:
            self._heap = []
            self._seen = data.seen.copy()
            self._pending = {}
            self._inflight = {}
            max_id = 0
            for req in data.pending:
                heapq.heappush(self._heap, req)
                self._pending[req.request_id] = req
                if req.request_id > max_id:
                    max_id = req.request_id
            self._next_id = max_id

    def clear(self) -> None:
        """Clear all state."""
        with self._lock:
            self._heap.clear()
            self._seen.clear()
            self._pending.clear()
            self._inflight.clear()
            self._next_id = 0
