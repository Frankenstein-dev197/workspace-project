"""Tests for savings ledger."""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from daemon_engine.memory.savings_ledger import (
    SavingsLedger,
    SavingsEvent,
    SavingsBucket,
    SavingsReport,
    estimate_cost_usd,
    SCHEMA_VERSION,
    UNKNOWN,
    _utc_now,
    _coerce_timestamp,
    _normalize_model,
    _label,
)


class TestEstimateCostUsd:
    def test_basic(self):
        cost = estimate_cost_usd("gpt-4", 1000)
        assert cost > 0
        assert cost == round(1000 * 3.0 / 1_000_000, 6)

    def test_zero_tokens(self):
        assert estimate_cost_usd("gpt-4", 0) == 0.0

    def test_negative_tokens_clamped(self):
        assert estimate_cost_usd("gpt-4", -100) == 0.0

    def test_custom_rate(self):
        cost = estimate_cost_usd("model", 1000, fallback_rate=0.001)
        assert cost == round(1000 * 0.001, 6)


class TestNormalizeModel:
    def test_string(self):
        assert _normalize_model("GPT-4") == "gpt-4"

    def test_empty(self):
        assert _normalize_model("") == UNKNOWN

    def test_none(self):
        assert _normalize_model(None) == UNKNOWN

    def test_strips_whitespace(self):
        assert _normalize_model("  gpt-4  ") == "gpt-4"


class TestLabel:
    def test_string(self):
        assert _label("My App") == "my-app"

    def test_empty(self):
        assert _label("") == UNKNOWN

    def test_none(self):
        assert _label(None) == UNKNOWN

    def test_truncates(self):
        long = "x" * 100
        assert len(_label(long)) <= 64


class TestCoerceTimestamp:
    def test_datetime(self):
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert _coerce_timestamp(dt) == dt

    def test_naive_datetime_assumed_utc(self):
        dt = datetime(2024, 1, 1)
        result = _coerce_timestamp(dt)
        assert result.tzinfo == timezone.utc

    def test_iso_string(self):
        result = _coerce_timestamp("2024-01-01T00:00:00+00:00")
        assert result.year == 2024

    def test_invalid_string(self):
        result = _coerce_timestamp("not-a-date")
        assert result.tzinfo == timezone.utc

    def test_none(self):
        result = _coerce_timestamp(None)
        assert result.tzinfo == timezone.utc


class TestSavingsEvent:
    def test_creation(self):
        event = SavingsEvent(
            tokens_before=1000,
            tokens_after=600,
            model="gpt-4",
        )
        assert event.tokens_saved == 400

    def test_no_savings(self):
        event = SavingsEvent(tokens_before=100, tokens_after=100)
        assert event.tokens_saved == 0

    def test_negative_clamped(self):
        event = SavingsEvent(tokens_before=50, tokens_after=100)
        assert event.tokens_saved == 0

    def test_to_dict(self):
        event = SavingsEvent(
            tokens_before=1000,
            tokens_after=600,
            model="gpt-4",
            client="my-app",
            source="engine",
        )
        d = event.to_dict()
        assert d["v"] == SCHEMA_VERSION
        assert d["before"] == 1000
        assert d["after"] == 600
        assert d["saved"] == 400
        assert d["model"] == "gpt-4"
        assert d["client"] == "my-app"
        assert "ts" in d
        assert "pid" in d


class TestSavingsBucket:
    def test_creation(self):
        bucket = SavingsBucket()
        assert bucket.tokens_saved == 0
        assert bucket.calls == 0

    def test_add(self):
        bucket = SavingsBucket()
        bucket.add(saved=100, before=1000, cost=0.01)
        bucket.add(saved=200, before=2000, cost=0.02)
        assert bucket.tokens_saved == 300
        assert bucket.tokens_before == 3000
        assert bucket.cost_usd == 0.03
        assert bucket.calls == 2

    def test_savings_percent(self):
        bucket = SavingsBucket()
        bucket.add(saved=300, before=1000, cost=0.0)
        assert bucket.savings_percent == 30.0

    def test_savings_percent_zero_before(self):
        bucket = SavingsBucket()
        assert bucket.savings_percent == 0.0

    def test_to_dict(self):
        bucket = SavingsBucket()
        bucket.add(saved=100, before=1000, cost=0.01)
        d = bucket.to_dict()
        assert d["tokens_saved"] == 100
        assert d["tokens_before"] == 1000
        assert d["calls"] == 1
        assert d["savings_percent"] == 10.0


class TestSavingsLedger:
    def setup_method(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.ledger_path = os.path.join(self.tmpdir, "savings.jsonl")
        self.ledger = SavingsLedger(self.ledger_path)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_record_creates_file(self):
        result = self.ledger.record(
            tokens_before=1000,
            tokens_after=600,
            model="gpt-4",
        )
        assert result is True
        assert os.path.exists(self.ledger_path)

    def test_record_no_savings_returns_false(self):
        result = self.ledger.record(
            tokens_before=100,
            tokens_after=100,
        )
        assert result is False

    def test_record_negative_after(self):
        result = self.ledger.record(
            tokens_before=100,
            tokens_after=200,
        )
        assert result is False

    def test_record_invalid_tokens(self):
        result = self.ledger.record(
            tokens_before="not-a-number",
            tokens_after=100,
        )
        assert result is False

    def test_record_writes_jsonl(self):
        self.ledger.record(tokens_before=1000, tokens_after=600, model="gpt-4")
        with open(self.ledger_path) as f:
            line = f.readline()
        event = json.loads(line)
        assert event["before"] == 1000
        assert event["after"] == 600
        assert event["saved"] == 400
        assert event["model"] == "gpt-4"
        assert "cost_usd" in event

    def test_record_multiple_events(self):
        for i in range(5):
            self.ledger.record(
                tokens_before=1000,
                tokens_after=600,
                model=f"model-{i % 2}",
                client=f"client-{i % 3}",
            )
        with open(self.ledger_path) as f:
            lines = f.readlines()
        assert len(lines) == 5

    def test_record_with_explicit_cost(self):
        self.ledger.record(
            tokens_before=1000,
            tokens_after=600,
            cost_usd=0.05,
        )
        with open(self.ledger_path) as f:
            event = json.loads(f.readline())
        assert event["cost_usd"] == 0.05

    def test_record_with_source(self):
        self.ledger.record(
            tokens_before=1000,
            tokens_after=600,
            source="compression",
        )
        with open(self.ledger_path) as f:
            event = json.loads(f.readline())
        assert event["source"] == "compression"

    def test_record_with_timestamp(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        self.ledger.record(
            tokens_before=1000,
            tokens_after=600,
            timestamp=ts,
        )
        with open(self.ledger_path) as f:
            event = json.loads(f.readline())
        assert event["ts"].startswith("2024-01-01")

    def test_aggregate_empty(self):
        report = self.ledger.aggregate()
        assert report.lifetime["tokens_saved"] == 0
        assert report.lifetime["calls"] == 0
        assert report.top_model == UNKNOWN

    def test_aggregate_single_event(self):
        self.ledger.record(
            tokens_before=1000,
            tokens_after=600,
            model="gpt-4",
            client="my-app",
        )
        report = self.ledger.aggregate()
        assert report.lifetime["tokens_saved"] == 400
        assert report.lifetime["calls"] == 1
        assert report.top_model == "gpt-4"

    def test_aggregate_multiple_events(self):
        for _ in range(3):
            self.ledger.record(
                tokens_before=1000,
                tokens_after=600,
                model="gpt-4",
                client="app1",
            )
        for _ in range(2):
            self.ledger.record(
                tokens_before=2000,
                tokens_after=1500,
                model="claude",
                client="app2",
            )
        report = self.ledger.aggregate()
        assert report.lifetime["tokens_saved"] == 400 * 3 + 500 * 2
        assert report.lifetime["calls"] == 5
        assert len(report.by_model) == 2
        assert len(report.by_client) == 2

    def test_aggregate_windows_today(self):
        # Recent event
        self.ledger.record(
            tokens_before=1000,
            tokens_after=600,
            model="gpt-4",
        )
        report = self.ledger.aggregate()
        assert report.windows["today"]["tokens_saved"] == 400
        assert report.windows["last_7_days"]["tokens_saved"] == 400

    def test_aggregate_by_model_ranked(self):
        self.ledger.record(
            tokens_before=1000, tokens_after=600, model="gpt-4", client="c1"
        )
        self.ledger.record(
            tokens_before=10000, tokens_after=5000, model="claude", client="c2"
        )
        report = self.ledger.aggregate()
        assert report.by_model[0]["model"] == "claude"  # more saved

    def test_aggregate_savings_percent(self):
        self.ledger.record(
            tokens_before=1000,
            tokens_after=600,
            model="gpt-4",
        )
        report = self.ledger.aggregate()
        assert report.lifetime["savings_percent"] == 40.0

    def test_aggregate_to_dict(self):
        self.ledger.record(tokens_before=1000, tokens_after=600, model="gpt-4")
        report = self.ledger.aggregate()
        d = report.to_dict()
        assert "schema_version" in d
        assert "lifetime" in d
        assert "windows" in d
        assert "by_model" in d

    def test_clear(self):
        self.ledger.record(tokens_before=1000, tokens_after=600)
        assert os.path.exists(self.ledger_path)
        self.ledger.clear()
        assert not os.path.exists(self.ledger_path)

    def test_thread_safe_concurrent_writes(self):
        import threading

        def writer(i):
            for j in range(20):
                self.ledger.record(
                    tokens_before=1000,
                    tokens_after=600,
                    model=f"model-{i}",
                )

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with open(self.ledger_path) as f:
            lines = f.readlines()
        assert len(lines) == 100

    def test_retention_filters_old_events(self):
        # Write an event with an old timestamp
        old_ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
        self.ledger.record(
            tokens_before=1000,
            tokens_after=600,
            timestamp=old_ts,
        )
        # Should be filtered out by retention
        report = self.ledger.aggregate()
        assert report.lifetime["calls"] == 0

    def test_retention_keeps_recent(self):
        recent_ts = datetime.now(timezone.utc) - timedelta(days=1)
        self.ledger.record(
            tokens_before=1000,
            tokens_after=600,
            timestamp=recent_ts,
        )
        report = self.ledger.aggregate()
        assert report.lifetime["calls"] == 1
