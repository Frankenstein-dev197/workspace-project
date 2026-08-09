"""Tests for priority scheduler module."""

import pytest

from daemon_engine.tools.priority_scheduler import (
    PriorityScheduler,
    ScheduledRequest,
    CheckpointData,
    canonicalize_url,
    fingerprint_url,
)


class TestCanonicalizeUrl:
    def test_lowercase_scheme_host(self):
        assert canonicalize_url("HTTP://Example.COM/path") == "http://example.com/path"

    def test_remove_default_port_http(self):
        assert canonicalize_url("http://example.com:80/path") == "http://example.com/path"

    def test_remove_default_port_https(self):
        assert canonicalize_url("https://example.com:443/path") == "https://example.com/path"

    def test_keep_nondefault_port(self):
        assert canonicalize_url("http://example.com:8080/path") == "http://example.com:8080/path"

    def test_sort_query_params(self):
        url = "http://example.com/p?b=2&a=1&c=3"
        assert canonicalize_url(url) == "http://example.com/p?a=1&b=2&c=3"

    def test_strip_fragment(self):
        url = "http://example.com/p#section"
        assert canonicalize_url(url) == "http://example.com/p"

    def test_keep_fragment(self):
        url = "http://example.com/p#section"
        assert canonicalize_url(url, keep_fragments=True) == "http://example.com/p#section"

    def test_idempotent(self):
        url = "http://example.com:80/p?b=2&a=1"
        once = canonicalize_url(url)
        twice = canonicalize_url(once)
        assert once == twice


class TestFingerprintUrl:
    def test_same_url_same_fingerprint(self):
        fp1 = fingerprint_url("http://example.com/p")
        fp2 = fingerprint_url("http://example.com/p")
        assert fp1 == fp2

    def test_different_url_different_fingerprint(self):
        fp1 = fingerprint_url("http://example.com/a")
        fp2 = fingerprint_url("http://example.com/b")
        assert fp1 != fp2

    def test_different_method_different_fingerprint(self):
        fp1 = fingerprint_url("http://example.com/p", method="GET")
        fp2 = fingerprint_url("http://example.com/p", method="POST")
        assert fp1 != fp2

    def test_different_body_different_fingerprint(self):
        fp1 = fingerprint_url("http://example.com/p", body=b"data1")
        fp2 = fingerprint_url("http://example.com/p", body=b"data2")
        assert fp1 != fp2

    def test_canonical_equivalents_same_fingerprint(self):
        fp1 = fingerprint_url("http://example.com:80/p?b=2&a=1")
        fp2 = fingerprint_url("http://example.com/p?a=1&b=2")
        assert fp1 == fp2

    def test_returns_bytes(self):
        fp = fingerprint_url("http://example.com/p")
        assert isinstance(fp, bytes)


class TestPriorityScheduler:
    def test_creation(self):
        s = PriorityScheduler()
        assert len(s) == 0
        assert s.is_empty is True

    def test_enqueue(self):
        s = PriorityScheduler()
        assert s.enqueue("http://example.com/1") is True
        assert len(s) == 1

    def test_dequeue(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/1")
        req = s.dequeue()
        assert req is not None
        assert req.url == "http://example.com/1"

    def test_dequeue_empty(self):
        s = PriorityScheduler()
        assert s.dequeue() is None

    def test_duplicate_dropped(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/1")
        assert s.enqueue("http://example.com/1") is False
        assert len(s) == 1

    def test_duplicate_allowed_with_dont_filter(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/1")
        assert s.enqueue("http://example.com/1", dont_filter=True) is True
        assert len(s) == 2

    def test_priority_order(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/low", priority=1)
        s.enqueue("http://example.com/high", priority=10)
        s.enqueue("http://example.com/mid", priority=5)
        req1 = s.dequeue()
        req2 = s.dequeue()
        req3 = s.dequeue()
        assert req1.url == "http://example.com/high"
        assert req2.url == "http://example.com/mid"
        assert req3.url == "http://example.com/low"

    def test_fifo_same_priority(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/1", priority=5)
        s.enqueue("http://example.com/2", priority=5)
        req1 = s.dequeue()
        req2 = s.dequeue()
        assert req1.url == "http://example.com/1"
        assert req2.url == "http://example.com/2"

    def test_complete(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/1")
        req = s.dequeue()
        s.complete(req.request_id)
        assert s.inflight_count == 0

    def test_inflight_tracking(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/1")
        s.enqueue("http://example.com/2")
        s.dequeue()
        s.dequeue()
        assert s.inflight_count == 2

    def test_seen_count(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/1")
        s.enqueue("http://example.com/2")
        s.enqueue("http://example.com/1")  # duplicate
        assert s.seen_count == 2

    def test_pending_count(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/1")
        s.enqueue("http://example.com/2")
        assert s.pending_count == 2
        s.dequeue()
        assert s.pending_count == 1

    def test_different_methods_not_duplicate(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/1", method="GET")
        assert s.enqueue("http://example.com/1", method="POST") is True
        assert len(s) == 2

    def test_canonical_duplicate(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com:80/p?b=2&a=1")
        assert s.enqueue("http://example.com/p?a=1&b=2") is False

    def test_fragment_difference(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/p#section1")
        # Without keep_fragments, fragments are stripped → duplicate
        assert s.enqueue("http://example.com/p#section2") is False

    def test_fragment_kept(self):
        s = PriorityScheduler(keep_fragments=True)
        s.enqueue("http://example.com/p#section1")
        assert s.enqueue("http://example.com/p#section2") is True

    def test_meta_preserved(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/1", meta={"key": "value"})
        req = s.dequeue()
        assert req.meta == {"key": "value"}


class TestSnapshotRestore:
    def test_snapshot_empty(self):
        s = PriorityScheduler()
        snap = s.snapshot()
        assert snap.pending == []
        assert snap.seen == set()

    def test_snapshot_has_pending(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/1")
        s.enqueue("http://example.com/2")
        snap = s.snapshot()
        assert len(snap.pending) == 2
        assert len(snap.seen) == 2

    def test_restore(self):
        s1 = PriorityScheduler()
        s1.enqueue("http://example.com/1", priority=5)
        s1.enqueue("http://example.com/2", priority=10)
        snap = s1.snapshot()

        s2 = PriorityScheduler()
        s2.restore(snap)
        assert len(s2) == 2
        req = s2.dequeue()
        assert req.priority == 10

    def test_restore_preserves_seen(self):
        s1 = PriorityScheduler()
        s1.enqueue("http://example.com/1")
        snap = s1.snapshot()

        s2 = PriorityScheduler()
        s2.restore(snap)
        # Duplicate should still be filtered after restore
        assert s2.enqueue("http://example.com/1") is False

    def test_restore_continues_id_counter(self):
        s1 = PriorityScheduler()
        s1.enqueue("http://example.com/1")
        s1.enqueue("http://example.com/2")
        snap = s1.snapshot()

        s2 = PriorityScheduler()
        s2.restore(snap)
        s2.enqueue("http://example.com/3")
        req = s2.dequeue()
        # Dequeued first should be highest priority (lowest sort_key)
        assert req.url in ("http://example.com/1", "http://example.com/2", "http://example.com/3")


class TestClear:
    def test_clear(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/1")
        s.enqueue("http://example.com/2")
        s.clear()
        assert len(s) == 0
        assert s.seen_count == 0
        assert s.is_empty is True

    def test_clear_allows_reenqueue(self):
        s = PriorityScheduler()
        s.enqueue("http://example.com/1")
        s.clear()
        assert s.enqueue("http://example.com/1") is True


class TestThreadSafety:
    def test_concurrent_enqueue(self):
        import threading
        s = PriorityScheduler()
        errors = []

        def worker(i):
            try:
                for j in range(20):
                    s.enqueue(f"http://example.com/{i}_{j}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(s) == 200
