"""Tests for dedupe store."""

import time
import threading

import pytest

from daemon_engine.core.dedupe_store import (
    MemoryDedupeStore,
    DedupeStore,
    make_dedupe_key,
    DEFAULT_DEDUPE_TTL_SECONDS,
    DEFAULT_DEDUPE_MAX_ENTRIES,
)


class TestMakeDedupeKey:
    def test_from_parts(self):
        key = make_dedupe_key("slack", "ws1", "chat1", "msg1")
        assert key == ("slack", "ws1", "chat1", "msg1")

    def test_single_part(self):
        assert make_dedupe_key("only") == ("only",)

    def test_mixed_types(self):
        key = make_dedupe_key("a", 1, 2.0, None)
        assert key == ("a", 1, 2.0, None)


class TestMemoryDedupeStore:
    def test_creation_defaults(self):
        store = MemoryDedupeStore()
        assert store.count() == 0
        assert store._ttl == DEFAULT_DEDUPE_TTL_SECONDS
        assert store._max == DEFAULT_DEDUPE_MAX_ENTRIES

    def test_creation_custom(self):
        store = MemoryDedupeStore(ttl_seconds=60, max_entries=100)
        assert store._ttl == 60
        assert store._max == 100

    def test_try_record_new(self):
        store = MemoryDedupeStore()
        assert store.try_record(("k1",)) is False
        assert store.count() == 1

    def test_try_record_duplicate(self):
        store = MemoryDedupeStore()
        store.try_record(("k1",))
        assert store.try_record(("k1",)) is True
        assert store.count() == 1

    def test_release(self):
        store = MemoryDedupeStore()
        store.try_record(("k1",))
        store.release(("k1",))
        assert store.count() == 0

    def test_release_nonexistent(self):
        store = MemoryDedupeStore()
        store.release(("nonexistent",))  # should not raise

    def test_release_then_re_record(self):
        store = MemoryDedupeStore()
        store.try_record(("k1",))
        store.release(("k1",))
        assert store.try_record(("k1",)) is False

    def test_clear(self):
        store = MemoryDedupeStore()
        store.try_record(("k1",))
        store.try_record(("k2",))
        store.clear()
        assert store.count() == 0

    def test_contains(self):
        store = MemoryDedupeStore()
        store.try_record(("k1",))
        assert store.contains(("k1",)) is True
        assert store.contains(("k2",)) is False

    def test_evict_expired(self):
        store = MemoryDedupeStore(ttl_seconds=0.1)
        store.try_record(("k1",))
        time.sleep(0.2)
        store.try_record(("k2",))
        assert not store.contains(("k1",))
        assert store.contains(("k2",))

    def test_evict_overflow(self):
        store = MemoryDedupeStore(max_entries=3)
        store.try_record(("k1",))
        store.try_record(("k2",))
        store.try_record(("k3",))
        store.try_record(("k4",))
        assert store.count() == 3
        assert not store.contains(("k1",))
        assert store.contains(("k4",))

    def test_stats(self):
        store = MemoryDedupeStore()
        store.try_record(("k1",))
        store.try_record(("k1",))
        store.release(("k1",))
        stats = store.stats()
        assert stats["total_recorded"] == 1
        assert stats["total_duplicates"] == 1
        assert stats["total_released"] == 1

    def test_thread_safety(self):
        store = MemoryDedupeStore(max_entries=10000)
        results = []
        threads = []

        def worker(tid):
            for i in range(100):
                results.append(store.try_record((tid, i)))

        for t in range(10):
            threads.append(threading.Thread(target=worker, args=(t,)))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert store.count() == 1000
        assert all(r is False for r in results)

    def test_thread_safety_duplicates(self):
        store = MemoryDedupeStore()
        results = []
        threads = []

        def worker():
            results.append(store.try_record(("same-key",)))

        for _ in range(10):
            threads.append(threading.Thread(target=worker))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        false_count = results.count(False)
        true_count = results.count(True)
        assert false_count == 1
        assert true_count == 9

    def test_protocol_compliance(self):
        store: DedupeStore = MemoryDedupeStore()
        assert store.try_record(("k1",)) is False
        store.release(("k1",))
        store.clear()
        assert store.count() == 0
