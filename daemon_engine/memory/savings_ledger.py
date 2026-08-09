"""Savings ledger: durable JSONL record of token/size savings events.

Integrates Headroom savings_ledger pattern:
- record_savings_event: append one savings event to a durable JSONL ledger
  - Never raises; returns True/False
  - Computes cost_usd at write time (no historical drift on price changes)
  - Optional fcntl.flock for concurrent-safe append (Unix only)
  - Auto-compaction when file grows past threshold
- aggregate_savings: roll up events into lifetime / windowed / per-dimension views
  - Today / last 7 days / last 30 days windows
  - By model / by client breakdowns
  - Hard-capped retention (default 30 days)
- SavingsReport / SavingsBucket: aggregated report structures
- estimate_cost_usd: dollar value of saved tokens (with fallback rate)

Cost is computed at write time so historical numbers do not drift if
model pricing changes later. Useful for tracking context-compression
savings, memory deduplication savings, etc.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

fcntl: Any | None = None
try:
    import fcntl as _fcntl

    fcntl = _fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

SCHEMA_VERSION = 1
UNKNOWN = "unknown"

MAX_RETENTION_DAYS = 30
DEFAULT_RETENTION_DAYS = 30
DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN = 3.0 / 1_000_000

_COMPACT_SIZE_BYTES = 1 * 1024 * 1024


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        aware = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return aware.astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    return _utc_now()


def _normalize_model(model: Any) -> str:
    if not model:
        return UNKNOWN
    s = str(model).strip().lower()
    return s or UNKNOWN


def _label(value: Any) -> str:
    cleaned = str(value or "").strip().lower().replace(" ", "-")[:64]
    return cleaned or UNKNOWN


def estimate_cost_usd(
    model: str,
    tokens_saved: int,
    *,
    fallback_rate: float = DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN,
) -> float:
    """Dollar value of saved input tokens.

    Uses a fallback per-token rate. In production, this would integrate
    with litellm pricing; here we use a blended rate for portability.
    """
    rate = fallback_rate
    saved = max(int(tokens_saved), 0)
    return round(saved * rate, 6)


@dataclass
class SavingsEvent:
    """A single savings event."""
    tokens_before: int
    tokens_after: int
    model: str = UNKNOWN
    client: str = UNKNOWN
    source: str = "engine"
    cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=_utc_now)

    @property
    def tokens_saved(self) -> int:
        return max(self.tokens_before - self.tokens_after, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": SCHEMA_VERSION,
            "ts": self.timestamp.isoformat(),
            "before": self.tokens_before,
            "after": self.tokens_after,
            "saved": self.tokens_saved,
            "cost_usd": round(self.cost_usd, 6),
            "model": self.model,
            "client": self.client,
            "source": self.source,
            "pid": os.getpid(),
        }


@dataclass
class SavingsBucket:
    """Aggregated savings for a dimension (model, client, window)."""
    tokens_saved: int = 0
    tokens_before: int = 0
    cost_usd: float = 0.0
    calls: int = 0

    def add(self, *, saved: int, before: int, cost: float) -> None:
        self.tokens_saved += saved
        self.tokens_before += before
        self.cost_usd += cost
        self.calls += 1

    @property
    def savings_percent(self) -> float:
        if self.tokens_before <= 0:
            return 0.0
        return round(self.tokens_saved / self.tokens_before * 100, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_saved": self.tokens_saved,
            "tokens_before": self.tokens_before,
            "cost_usd": round(self.cost_usd, 6),
            "calls": self.calls,
            "savings_percent": self.savings_percent,
        }


@dataclass
class SavingsReport:
    """Aggregated savings report."""
    path: str
    schema_version: int
    lifetime: dict[str, Any]
    windows: dict[str, dict[str, Any]]
    by_model: list[dict[str, Any]]
    by_client: list[dict[str, Any]]
    top_model: str = UNKNOWN

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "path": self.path,
            "top_model": self.top_model,
            "lifetime": self.lifetime,
            "windows": self.windows,
            "by_model": self.by_model,
            "by_client": self.by_client,
        }


def _ranked(buckets: dict[str, SavingsBucket], key_name: str) -> list[dict[str, Any]]:
    rows = []
    for name, bucket in buckets.items():
        row = {key_name: name, **bucket.to_dict()}
        rows.append(row)
    rows.sort(key=lambda r: (r["cost_usd"], r["tokens_saved"]), reverse=True)
    return rows


class SavingsLedger:
    """Durable JSONL ledger of savings events.

    Thread-safe and process-safe (via fcntl on Unix). Never raises on
    write failures; returns True/False instead.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        fallback_rate: float = DEFAULT_FALLBACK_INPUT_COST_PER_TOKEN,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        self.path = Path(path)
        self.fallback_rate = fallback_rate
        self.retention_days = max(
            1, min(retention_days or MAX_RETENTION_DAYS, MAX_RETENTION_DAYS)
        )
        self._lock = threading.Lock()

    def record(
        self,
        *,
        tokens_before: int,
        tokens_after: int,
        model: Any = None,
        client: Any = None,
        source: str = "engine",
        timestamp: Any = None,
        cost_usd: float | None = None,
    ) -> bool:
        """Append one savings event to the durable ledger. Never raises.

        Returns True when a line was written. cost_usd is computed from
        the model + tokens saved when not supplied by the caller.
        """
        try:
            before = max(int(tokens_before), 0)
            after = max(int(tokens_after), 0)
        except (TypeError, ValueError):
            return False

        saved = max(before - after, 0)
        if saved <= 0:
            return False

        model_label = _normalize_model(model)
        if cost_usd is None:
            cost = estimate_cost_usd(
                model_label, saved, fallback_rate=self.fallback_rate
            )
        else:
            try:
                cost = max(float(cost_usd), 0.0)
            except (TypeError, ValueError):
                cost = 0.0

        event = {
            "v": SCHEMA_VERSION,
            "ts": _coerce_timestamp(timestamp).isoformat(),
            "before": before,
            "after": after,
            "saved": saved,
            "cost_usd": round(cost, 6),
            "model": model_label,
            "client": _label(client),
            "source": str(source or UNKNOWN),
            "pid": os.getpid(),
        }

        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                line = json.dumps(event, separators=(",", ":")) + "\n"
                with open(self.path, "a", encoding="utf-8") as handle:
                    if _HAS_FCNTL and fcntl is not None:
                        fcntl.flock(handle, fcntl.LOCK_EX)
                    try:
                        handle.write(line)
                    finally:
                        if _HAS_FCNTL and fcntl is not None:
                            fcntl.flock(handle, fcntl.LOCK_UN)
            except Exception:
                return False

            self._maybe_compact()
        return True

    def _read_events(
        self,
        *,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Read events within retention window."""
        now = now or _utc_now()
        cutoff = now - timedelta(days=self.retention_days)
        events = []
        if not self.path.exists():
            return events
        try:
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = _coerce_timestamp(event.get("ts"))
                    if ts < cutoff:
                        continue
                    event["_ts"] = ts
                    events.append(event)
        except Exception:
            return []
        return events

    def aggregate(
        self,
        *,
        now: datetime | None = None,
    ) -> SavingsReport:
        """Aggregate the durable ledger into lifetime/windowed/per-dim views."""
        now = now or _utc_now()
        events = self._read_events(now=now)

        today_cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_cutoff = now - timedelta(days=7)

        windowed = SavingsBucket()
        today = SavingsBucket()
        last_7 = SavingsBucket()
        by_model: dict[str, SavingsBucket] = {}
        by_client: dict[str, SavingsBucket] = {}

        for event in events:
            ts: datetime = event["_ts"]
            saved = max(int(event.get("saved", 0) or 0), 0)
            before = max(int(event.get("before", 0) or 0), 0)
            try:
                cost = max(float(event.get("cost_usd", 0.0) or 0.0), 0.0)
            except (TypeError, ValueError):
                cost = 0.0

            windowed.add(saved=saved, before=before, cost=cost)
            if ts >= today_cutoff:
                today.add(saved=saved, before=before, cost=cost)
            if ts >= week_cutoff:
                last_7.add(saved=saved, before=before, cost=cost)

            by_model.setdefault(
                str(event.get("model") or UNKNOWN), SavingsBucket()
            ).add(saved=saved, before=before, cost=cost)
            by_client.setdefault(
                str(event.get("client") or UNKNOWN), SavingsBucket()
            ).add(saved=saved, before=before, cost=cost)

        model_rows = _ranked(by_model, "model")
        top_model = model_rows[0]["model"] if model_rows else UNKNOWN

        return SavingsReport(
            path=str(self.path),
            schema_version=SCHEMA_VERSION,
            lifetime=windowed.to_dict(),
            windows={
                "today": today.to_dict(),
                "last_7_days": last_7.to_dict(),
                "last_30_days": windowed.to_dict(),
            },
            by_model=model_rows,
            by_client=_ranked(by_client, "client"),
            top_model=top_model,
        )

    def _maybe_compact(self) -> None:
        """Rewrite the ledger dropping out-of-retention events if it grows large."""
        try:
            if not self.path.exists():
                return
            if self.path.stat().st_size <= _COMPACT_SIZE_BYTES:
                return
            now = _utc_now()
            cutoff = now - timedelta(days=self.retention_days)
            kept = []
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = _coerce_timestamp(event.get("ts"))
                    if ts >= cutoff:
                        kept.append(line)
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as handle:
                for line in kept:
                    handle.write(line + "\n")
            tmp.replace(self.path)
        except Exception:
            pass

    def clear(self) -> None:
        """Remove all events from the ledger."""
        with self._lock:
            try:
                if self.path.exists():
                    self.path.unlink()
            except Exception:
                pass
