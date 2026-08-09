"""Cron scheduler: time-based job scheduling with cron expressions.

Integrates learn-claude-code s14 cron scheduler pattern:
- CronJob: id, cron expression, prompt, recurring, durable
- cron_matches: 5-field cron expression matching with DOM/DOW OR semantics
- validate_cron: validation with bounds checking
- schedule_job / cancel_job: register/remove jobs
- Thread-safe queue: scheduler writes, consumer reads
- Durable storage: persists to JSON file (survives restart)
- Date-aware firing: prevents daily jobs from skipping

Cron expression format: "minute hour day-of-month month day-of-week"
- minute: 0-59
- hour: 0-23
- day-of-month: 1-31
- month: 1-12
- day-of-week: 0-6 (0=Sunday)
- Supports: *, */N, ranges (1-5), lists (1,3,5)
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CRON_BOUNDS = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
CRON_NAMES = ["minute", "hour", "day-of-month", "month", "day-of-week"]


@dataclass
class CronJob:
    """A scheduled cron job."""
    id: str
    cron: str
    prompt: str
    recurring: bool = True
    durable: bool = True
    created_at: float = field(default_factory=time.time)
    last_fired: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cron_field_matches(field: str, value: int) -> bool:
    """Match a single cron field against a value."""
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    if "," in field:
        return any(_cron_field_matches(f.strip(), value) for f in field.split(","))
    if "-" in field:
        lo, hi = field.split("-", 1)
        return int(lo) <= value <= int(hi)
    return value == int(field)


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """Check if a 5-field cron expression matches the given datetime.

    Standard cron semantics: DOM and DOW use OR when both are constrained.
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields
    dow_val = (dt.weekday() + 1) % 7  # Python Monday=0 → cron Sunday=0

    m = _cron_field_matches(minute, dt.minute)
    h = _cron_field_matches(hour, dt.hour)
    dom_ok = _cron_field_matches(dom, dt.day)
    month_ok = _cron_field_matches(month, dt.month)
    dow_ok = _cron_field_matches(dow, dow_val)

    if not (m and h and month_ok):
        return False

    dom_unconstrained = dom == "*"
    dow_unconstrained = dow == "*"
    if dom_unconstrained and dow_unconstrained:
        return True
    if dom_unconstrained:
        return dow_ok
    if dow_unconstrained:
        return dom_ok
    return dom_ok or dow_ok


def _validate_cron_field(field: str, lo: int, hi: int) -> str | None:
    """Validate a single cron field. Returns error message or None."""
    if field == "*":
        return None
    if field.startswith("*/"):
        step_str = field[2:]
        if not step_str.isdigit():
            return f"Invalid step: {field}"
        step = int(step_str)
        if step <= 0:
            return f"Step must be > 0: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), lo, hi)
            if err:
                return err
        return None
    if "-" in field:
        parts = field.split("-", 1)
        if not parts[0].isdigit() or not parts[1].isdigit():
            return f"Invalid range: {field}"
        a, b = int(parts[0]), int(parts[1])
        if a < lo or a > hi or b < lo or b > hi:
            return f"Range {field} out of bounds [{lo}-{hi}]"
        if a > b:
            return f"Range start > end: {field}"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    val = int(field)
    if val < lo or val > hi:
        return f"Value {val} out of bounds [{lo}-{hi}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    """Validate a cron expression. Returns error message or None."""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"
    for i, (field, (lo, hi), name) in enumerate(zip(fields, CRON_BOUNDS, CRON_NAMES)):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None


class CronScheduler:
    """Thread-safe cron scheduler with durable storage.

    The scheduler maintains:
    - scheduled_jobs: dict of all registered jobs
    - fired_queue: jobs that have matched and are waiting to be consumed
    - _last_fired: tracks last fire time per job (prevents re-firing)
    """

    def __init__(self, durable_path: str | Path | None = None) -> None:
        self._durable_path = Path(durable_path) if durable_path else None
        self._jobs: dict[str, CronJob] = {}
        self._fired_queue: list[CronJob] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None
        self._stats = {
            "total_scheduled": 0,
            "total_fired": 0,
            "total_cancelled": 0,
            "active_jobs": 0,
        }
        if self._durable_path:
            self._load_durable()

    def schedule_job(
        self,
        cron: str,
        prompt: str,
        recurring: bool = True,
        durable: bool = True,
        job_id: str | None = None,
    ) -> CronJob | str:
        """Register a new cron job. Returns CronJob or error string."""
        err = validate_cron(cron)
        if err:
            return err
        job = CronJob(
            id=job_id or f"cron_{random.randint(0, 999999):06d}",
            cron=cron,
            prompt=prompt,
            recurring=recurring,
            durable=durable,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._stats["total_scheduled"] += 1
            self._stats["active_jobs"] = len(self._jobs)
        if durable:
            self._save_durable()
        logger.info("Cron job scheduled: %s '%s'", job.id, cron)
        return job

    def cancel_job(self, job_id: str) -> str:
        """Cancel a cron job."""
        with self._lock:
            job = self._jobs.pop(job_id, None)
            self._stats["active_jobs"] = len(self._jobs)
        if not job:
            return f"Job {job_id} not found"
        self._stats["total_cancelled"] += 1
        if job.durable:
            self._save_durable()
        logger.info("Cron job cancelled: %s", job_id)
        return f"Cancelled {job_id}"

    def list_jobs(self) -> list[CronJob]:
        """List all scheduled jobs."""
        with self._lock:
            return list(self._jobs.values())

    def get_job(self, job_id: str) -> CronJob | None:
        """Get a specific job."""
        with self._lock:
            return self._jobs.get(job_id)

    def check_and_fire(self, dt: datetime | None = None) -> list[CronJob]:
        """Check all jobs against the given datetime and fire matching ones.

        This is the core matching logic. Can be called manually or by
        the scheduler thread.
        """
        now = dt or datetime.now()
        minute_marker = now.strftime("%Y-%m-%d %H:%M")
        fired: list[CronJob] = []
        with self._lock:
            for job in list(self._jobs.values()):
                try:
                    if cron_matches(job.cron, now):
                        if job.last_fired != minute_marker:
                            self._fired_queue.append(job)
                            job.last_fired = minute_marker
                            fired.append(job)
                            self._stats["total_fired"] += 1
                            if not job.recurring:
                                self._jobs.pop(job.id, None)
                except Exception as exc:
                    logger.error("Cron job %s error: %s", job.id, exc)
        if fired and any(j.durable for j in fired):
            self._save_durable()
        return fired

    def consume_fired(self) -> list[CronJob]:
        """Consume fired jobs from the queue (called by consumer)."""
        with self._lock:
            fired = list(self._fired_queue)
            self._fired_queue.clear()
        return fired

    def has_fired(self) -> bool:
        """Return whether fired jobs are waiting to be consumed."""
        with self._lock:
            return bool(self._fired_queue)

    def start(self, poll_interval: float = 1.0) -> None:
        """Start the scheduler daemon thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._scheduler_loop,
            args=(poll_interval,),
            daemon=True,
        )
        self._thread.start()
        logger.info("Cron scheduler started")

    def stop(self) -> None:
        """Stop the scheduler thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        logger.info("Cron scheduler stopped")

    def _scheduler_loop(self, poll_interval: float) -> None:
        """Daemon thread: poll every interval, fire matching jobs."""
        while self._running:
            time.sleep(poll_interval)
            self.check_and_fire()

    def _save_durable(self) -> None:
        """Persist durable jobs to disk."""
        if not self._durable_path:
            return
        with self._lock:
            durable = [j.to_dict() for j in self._jobs.values() if j.durable]
        try:
            self._durable_path.parent.mkdir(parents=True, exist_ok=True)
            self._durable_path.write_text(json.dumps(durable, indent=2))
        except Exception as exc:
            logger.error("Failed to save durable jobs: %s", exc)

    def _load_durable(self) -> None:
        """Load durable jobs from disk on startup."""
        if not self._durable_path or not self._durable_path.exists():
            return
        try:
            jobs = json.loads(self._durable_path.read_text())
            for j in jobs:
                job = CronJob(**j)
                err = validate_cron(job.cron)
                if err:
                    logger.warning("Skipping invalid job %s: %s", job.id, err)
                    continue
                self._jobs[job.id] = job
            self._stats["active_jobs"] = len(self._jobs)
            logger.info("Loaded %d durable cron jobs", len(self._jobs))
        except Exception as exc:
            logger.error("Failed to load durable jobs: %s", exc)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "active_jobs": len(self._jobs),
                "fired_pending": len(self._fired_queue),
                "running": self._running,
            }

    def clear(self) -> None:
        """Clear all jobs and fired queue."""
        with self._lock:
            self._jobs.clear()
            self._fired_queue.clear()
            self._stats["active_jobs"] = 0
        if self._durable_path and self._durable_path.exists():
            self._durable_path.unlink()
