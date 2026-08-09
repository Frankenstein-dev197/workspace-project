"""Tests for cron scheduler."""

import json
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pytest

from daemon_engine.runtime.cron_scheduler import (
    CronScheduler,
    CronJob,
    cron_matches,
    validate_cron,
    _cron_field_matches,
    _validate_cron_field,
)


class TestCronFieldMatching:
    def test_star_matches_anything(self):
        assert _cron_field_matches("*", 5) is True

    def test_exact_value(self):
        assert _cron_field_matches("5", 5) is True
        assert _cron_field_matches("5", 6) is False

    def test_step(self):
        assert _cron_field_matches("*/15", 0) is True
        assert _cron_field_matches("*/15", 15) is True
        assert _cron_field_matches("*/15", 30) is True
        assert _cron_field_matches("*/15", 7) is False

    def test_range(self):
        assert _cron_field_matches("1-5", 1) is True
        assert _cron_field_matches("1-5", 3) is True
        assert _cron_field_matches("1-5", 5) is True
        assert _cron_field_matches("1-5", 6) is False

    def test_list(self):
        assert _cron_field_matches("1,3,5", 1) is True
        assert _cron_field_matches("1,3,5", 3) is True
        assert _cron_field_matches("1,3,5", 5) is True
        assert _cron_field_matches("1,3,5", 2) is False


class TestCronMatches:
    def test_every_minute(self):
        dt = datetime(2024, 1, 15, 10, 30)
        assert cron_matches("* * * * *", dt) is True

    def test_specific_time(self):
        dt = datetime(2024, 1, 15, 10, 30)
        assert cron_matches("30 10 * * *", dt) is True
        assert cron_matches("31 10 * * *", dt) is False

    def test_specific_hour_and_minute(self):
        dt = datetime(2024, 1, 15, 14, 30)
        assert cron_matches("30 14 * * *", dt) is True
        assert cron_matches("30 15 * * *", dt) is False

    def test_day_of_month(self):
        dt = datetime(2024, 1, 15, 10, 30)
        assert cron_matches("30 10 15 * *", dt) is True
        assert cron_matches("30 10 16 * *", dt) is False

    def test_month(self):
        dt = datetime(2024, 1, 15, 10, 30)
        assert cron_matches("30 10 15 1 *", dt) is True
        assert cron_matches("30 10 15 2 *", dt) is False

    def test_day_of_week_sunday(self):
        dt = datetime(2024, 1, 7, 10, 30)  # Sunday
        assert cron_matches("30 10 * * 0", dt) is True

    def test_day_of_week_monday(self):
        dt = datetime(2024, 1, 1, 10, 30)  # Monday
        assert cron_matches("30 10 * * 1", dt) is True

    def test_dom_dow_or_semantics(self):
        dt = datetime(2024, 1, 15, 10, 30)  # Monday
        assert cron_matches("30 10 15 * 1", dt) is True
        dt2 = datetime(2024, 1, 16, 10, 30)  # Tuesday, DOM=16
        assert cron_matches("30 10 15 * 1", dt2) is False

    def test_invalid_expression(self):
        dt = datetime(2024, 1, 15, 10, 30)
        assert cron_matches("invalid", dt) is False
        assert cron_matches("* * *", dt) is False

    def test_step_minute(self):
        dt = datetime(2024, 1, 15, 10, 0)
        assert cron_matches("*/15 * * * *", dt) is True
        dt2 = datetime(2024, 1, 15, 10, 7)
        assert cron_matches("*/15 * * * *", dt2) is False


class TestValidateCron:
    def test_valid_expression(self):
        assert validate_cron("* * * * *") is None
        assert validate_cron("30 10 * * *") is None
        assert validate_cron("0 0 1 1 0") is None

    def test_wrong_field_count(self):
        assert validate_cron("* * *") is not None
        assert validate_cron("* * * * * *") is not None

    def test_out_of_bounds(self):
        assert validate_cron("60 * * * *") is not None  # minute > 59
        assert validate_cron("* 24 * * *") is not None  # hour > 23
        assert validate_cron("* * 32 * *") is not None  # dom > 31
        assert validate_cron("* * * 13 *") is not None  # month > 12
        assert validate_cron("* * * * 7") is not None   # dow > 6

    def test_invalid_step(self):
        assert validate_cron("*/0 * * * *") is not None

    def test_invalid_range(self):
        assert validate_cron("5-3 * * * *") is not None  # start > end

    def test_invalid_field(self):
        assert validate_cron("abc * * * *") is not None

    def test_valid_complex(self):
        assert validate_cron("*/15 9-17 * * 1-5") is None
        assert validate_cron("0,30 0 * * 0,6") is None


class TestCronScheduler:
    def test_creation(self):
        scheduler = CronScheduler()
        assert len(scheduler.list_jobs()) == 0

    def test_schedule_job(self):
        scheduler = CronScheduler()
        job = scheduler.schedule_job("* * * * *", "Test prompt")
        assert isinstance(job, CronJob)
        assert len(scheduler.list_jobs()) == 1

    def test_schedule_invalid_cron(self):
        scheduler = CronScheduler()
        result = scheduler.schedule_job("invalid", "Test")
        assert isinstance(result, str)

    def test_cancel_job(self):
        scheduler = CronScheduler()
        job = scheduler.schedule_job("* * * * *", "Test")
        result = scheduler.cancel_job(job.id)
        assert "Cancelled" in result
        assert len(scheduler.list_jobs()) == 0

    def test_cancel_nonexistent(self):
        scheduler = CronScheduler()
        result = scheduler.cancel_job("nonexistent")
        assert "not found" in result

    def test_check_and_fire(self):
        scheduler = CronScheduler()
        scheduler.schedule_job("* * * * *", "Test")
        dt = datetime(2024, 1, 15, 10, 30)
        fired = scheduler.check_and_fire(dt)
        assert len(fired) == 1

    def test_no_fire_on_mismatch(self):
        scheduler = CronScheduler()
        scheduler.schedule_job("0 0 * * *", "Midnight")
        dt = datetime(2024, 1, 15, 10, 30)
        fired = scheduler.check_and_fire(dt)
        assert len(fired) == 0

    def test_no_double_fire_same_minute(self):
        scheduler = CronScheduler()
        scheduler.schedule_job("* * * * *", "Test")
        dt = datetime(2024, 1, 15, 10, 30)
        scheduler.check_and_fire(dt)
        fired = scheduler.check_and_fire(dt)
        assert len(fired) == 0

    def test_consume_fired(self):
        scheduler = CronScheduler()
        scheduler.schedule_job("* * * * *", "Test")
        dt = datetime(2024, 1, 15, 10, 30)
        scheduler.check_and_fire(dt)
        assert scheduler.has_fired() is True
        consumed = scheduler.consume_fired()
        assert len(consumed) == 1
        assert scheduler.has_fired() is False

    def test_one_shot_job_removed_after_fire(self):
        scheduler = CronScheduler()
        scheduler.schedule_job("* * * * *", "Test", recurring=False)
        dt = datetime(2024, 1, 15, 10, 30)
        scheduler.check_and_fire(dt)
        assert len(scheduler.list_jobs()) == 0

    def test_recurring_job_persists(self):
        scheduler = CronScheduler()
        scheduler.schedule_job("* * * * *", "Test", recurring=True)
        dt = datetime(2024, 1, 15, 10, 30)
        scheduler.check_and_fire(dt)
        assert len(scheduler.list_jobs()) == 1

    def test_get_job(self):
        scheduler = CronScheduler()
        job = scheduler.schedule_job("* * * * *", "Test")
        retrieved = scheduler.get_job(job.id)
        assert retrieved is not None
        assert retrieved.id == job.id

    def test_durable_persistence(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            scheduler1 = CronScheduler(durable_path=path)
            scheduler1.schedule_job("0 0 * * *", "Daily task", durable=True)
            scheduler2 = CronScheduler(durable_path=path)
            assert len(scheduler2.list_jobs()) == 1
        finally:
            Path(path).unlink(missing_ok=True)

    def test_durable_skip_invalid(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            data = [{"id": "bad", "cron": "invalid", "prompt": "test", "recurring": True, "durable": True, "created_at": 0, "last_fired": None}]
            Path(path).write_text(json.dumps(data))
            scheduler = CronScheduler(durable_path=path)
            assert len(scheduler.list_jobs()) == 0
        finally:
            Path(path).unlink(missing_ok=True)

    def test_stats(self):
        scheduler = CronScheduler()
        scheduler.schedule_job("* * * * *", "Test")
        scheduler.check_and_fire(datetime(2024, 1, 15, 10, 30))
        stats = scheduler.stats()
        assert stats["total_scheduled"] == 1
        assert stats["total_fired"] == 1
        assert stats["active_jobs"] == 1

    def test_clear(self):
        scheduler = CronScheduler()
        scheduler.schedule_job("* * * * *", "Test")
        scheduler.clear()
        assert len(scheduler.list_jobs()) == 0

    def test_start_stop(self):
        scheduler = CronScheduler()
        scheduler.start(poll_interval=0.1)
        assert scheduler.stats()["running"] is True
        scheduler.stop()
        assert scheduler.stats()["running"] is False

    def test_custom_job_id(self):
        scheduler = CronScheduler()
        job = scheduler.schedule_job("* * * * *", "Test", job_id="custom_id")
        assert job.id == "custom_id"
