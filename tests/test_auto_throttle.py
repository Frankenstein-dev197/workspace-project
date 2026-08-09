"""Tests for auto-throttle module."""

import pytest

from daemon_engine.tools.auto_throttle import (
    AutoThrottle,
    parse_retry_after,
    DomainStats,
)


class TestParseRetryAfter:
    def test_numeric_seconds(self):
        headers = {"Retry-After": "120"}
        assert parse_retry_after(headers) == 120.0

    def test_case_insensitive(self):
        headers = {"retry-after": "60"}
        assert parse_retry_after(headers) == 60.0

    def test_missing(self):
        headers = {"Content-Type": "text/html"}
        assert parse_retry_after(headers) is None

    def test_empty(self):
        headers = {"Retry-After": ""}
        assert parse_retry_after(headers) is None

    def test_negative_clamped(self):
        headers = {"Retry-After": "-5"}
        assert parse_retry_after(headers) == 0.0

    def test_float(self):
        headers = {"Retry-After": "1.5"}
        assert parse_retry_after(headers) == 1.5

    def test_invalid_string(self):
        headers = {"Retry-After": "not-a-date"}
        assert parse_retry_after(headers) is None

    def test_http_date_future(self):
        from email.utils import format_datetime
        from datetime import datetime, timezone, timedelta
        future = datetime.now(timezone.utc) + timedelta(seconds=100)
        headers = {"Retry-After": format_datetime(future)}
        result = parse_retry_after(headers)
        assert result is not None
        assert 90 <= result <= 110

    def test_http_date_past_clamped(self):
        from email.utils import format_datetime
        from datetime import datetime, timezone, timedelta
        past = datetime.now(timezone.utc) - timedelta(seconds=100)
        headers = {"Retry-After": format_datetime(past)}
        assert parse_retry_after(headers) == 0.0


class TestAutoThrottleInit:
    def test_defaults(self):
        at = AutoThrottle()
        assert at.start_delay == 5.0
        assert at.max_delay == 60.0
        assert at.target_concurrency == 1.0
        assert at.block_backoff is True

    def test_custom(self):
        at = AutoThrottle(start_delay=1.0, max_delay=10.0, target_concurrency=2.0)
        assert at.start_delay == 1.0
        assert at.max_delay == 10.0

    def test_invalid_concurrency(self):
        with pytest.raises(ValueError):
            AutoThrottle(target_concurrency=0)

    def test_invalid_negative_concurrency(self):
        with pytest.raises(ValueError):
            AutoThrottle(target_concurrency=-1)

    def test_max_less_than_start(self):
        with pytest.raises(ValueError):
            AutoThrottle(start_delay=10.0, max_delay=5.0)


class TestAutoThrottleGetDelay:
    def test_default_for_unknown_domain(self):
        at = AutoThrottle(start_delay=3.0)
        assert at.get_delay("example.com") == 3.0

    def test_set_delay(self):
        at = AutoThrottle()
        at.set_delay("example.com", 10.0)
        assert at.get_delay("example.com") == 10.0

    def test_set_delay_clamped_to_max(self):
        at = AutoThrottle(max_delay=30.0)
        at.set_delay("example.com", 100.0)
        assert at.get_delay("example.com") == 30.0

    def test_set_delay_clamped_to_zero(self):
        at = AutoThrottle()
        at.set_delay("example.com", -5.0)
        assert at.get_delay("example.com") == 0.0


class TestAutoThrottleRecordRequest:
    def test_fast_response_reduces_delay(self):
        at = AutoThrottle(start_delay=4.0)
        at.record_request("example.com", latency=1.0)
        assert at.get_delay("example.com") < 4.0

    def test_slow_response_increases_delay(self):
        at = AutoThrottle(start_delay=1.0, max_delay=60.0)
        at.record_request("example.com", latency=10.0)
        assert at.get_delay("example.com") > 1.0

    def test_normal_response_keeps_delay(self):
        at = AutoThrottle(start_delay=5.0)
        new_delay = at.record_request("example.com", latency=5.0)
        assert new_delay == 5.0

    def test_blocked_doubles_delay(self):
        at = AutoThrottle(start_delay=5.0, max_delay=60.0)
        at.record_request("example.com", latency=1.0, blocked=True)
        assert at.get_delay("example.com") == 10.0

    def test_blocked_with_retry_after(self):
        at = AutoThrottle(start_delay=5.0, max_delay=60.0)
        new_delay = at.record_request(
            "example.com", latency=1.0, blocked=True, retry_after=30.0
        )
        assert new_delay == 30.0

    def test_blocked_with_short_retry_after_uses_current(self):
        at = AutoThrottle(start_delay=5.0, max_delay=60.0)
        at.set_delay("example.com", 20.0)
        new_delay = at.record_request(
            "example.com", latency=1.0, blocked=True, retry_after=5.0
        )
        assert new_delay == 20.0

    def test_blocked_clamped_to_max(self):
        at = AutoThrottle(start_delay=30.0, max_delay=40.0)
        at.record_request("example.com", latency=1.0, blocked=True)
        assert at.get_delay("example.com") == 40.0

    def test_block_backoff_disabled(self):
        at = AutoThrottle(start_delay=5.0, max_delay=60.0, block_backoff=False)
        at.record_request("example.com", latency=1.0, blocked=True)
        # Without backoff and no retry_after, delay stays at start
        assert at.get_delay("example.com") == 5.0

    def test_returns_new_delay(self):
        at = AutoThrottle(start_delay=5.0)
        new_delay = at.record_request("example.com", latency=1.0)
        assert new_delay == at.get_delay("example.com")


class TestAutoThrottleStats:
    def test_get_stats_unknown(self):
        at = AutoThrottle()
        assert at.get_stats("example.com") is None

    def test_get_stats_after_request(self):
        at = AutoThrottle()
        at.record_request("example.com", latency=2.0)
        stats = at.get_stats("example.com")
        assert stats is not None
        assert stats.total_requests == 1
        assert stats.last_latency == 2.0

    def test_stats_blocks(self):
        at = AutoThrottle()
        at.record_request("example.com", latency=1.0, blocked=True)
        at.record_request("example.com", latency=1.0, blocked=True)
        stats = at.get_stats("example.com")
        assert stats.total_blocks == 2

    def test_record_retry(self):
        at = AutoThrottle()
        at.record_retry("example.com")
        at.record_retry("example.com")
        stats = at.get_stats("example.com")
        assert stats.total_retries == 2

    def test_all_stats(self):
        at = AutoThrottle()
        at.record_request("a.com", latency=1.0)
        at.record_request("b.com", latency=2.0, blocked=True)
        all_stats = at.all_stats()
        assert "a.com" in all_stats
        assert "b.com" in all_stats
        assert all_stats["b.com"]["total_blocks"] == 1

    def test_all_stats_empty(self):
        at = AutoThrottle()
        assert at.all_stats() == {}


class TestAutoThrottleReset:
    def test_reset_domain(self):
        at = AutoThrottle(start_delay=5.0)
        at.set_delay("example.com", 20.0)
        at.reset_domain("example.com")
        assert at.get_delay("example.com") == 5.0

    def test_reset_unknown_domain(self):
        at = AutoThrottle()
        at.reset_domain("unknown.com")  # should not error

    def test_clear(self):
        at = AutoThrottle()
        at.set_delay("a.com", 10.0)
        at.set_delay("b.com", 20.0)
        at.clear()
        assert at.domain_count() == 0
        assert at.get_delay("a.com") == 5.0

    def test_domain_count(self):
        at = AutoThrottle()
        assert at.domain_count() == 0
        at.set_delay("a.com", 10.0)
        assert at.domain_count() == 1
        at.set_delay("b.com", 20.0)
        assert at.domain_count() == 2


class TestAutoThrottleThreadSafety:
    def test_concurrent_access(self):
        import threading
        at = AutoThrottle(start_delay=1.0, max_delay=60.0)
        errors = []

        def worker(domain):
            try:
                for _ in range(50):
                    at.record_request(domain, latency=0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"d{i}.com",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert at.domain_count() == 10
