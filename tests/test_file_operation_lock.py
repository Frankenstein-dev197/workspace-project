"""Tests for file operation locks."""

import threading
import time

import pytest

from daemon_engine.runtime.file_operation_lock import (
    FileOperationLockManager,
    get_file_operation_lock,
    get_file_operation_lock_key,
    acquire_file_operation_lock,
    active_lock_count,
    clear_locks,
)


class TestGetFileOperationLockKey:
    def test_basic(self):
        key = get_file_operation_lock_key("sandbox1", "/path/to/file")
        assert key == ("sandbox1", "/path/to/file")

    def test_empty_sandbox_defaults(self):
        key = get_file_operation_lock_key("", "/path")
        assert key == ("default", "/path")

    def test_different_paths_different_keys(self):
        k1 = get_file_operation_lock_key("s1", "/a")
        k2 = get_file_operation_lock_key("s1", "/b")
        assert k1 != k2

    def test_different_sandboxes_different_keys(self):
        k1 = get_file_operation_lock_key("s1", "/a")
        k2 = get_file_operation_lock_key("s2", "/a")
        assert k1 != k2


class TestGetFileOperationLock:
    def setup_method(self):
        clear_locks()

    def test_returns_lock(self):
        lock = get_file_operation_lock("s1", "/file")
        assert isinstance(lock, type(threading.Lock()))

    def test_same_key_returns_same_lock(self):
        lock1 = get_file_operation_lock("s1", "/file")
        lock2 = get_file_operation_lock("s1", "/file")
        assert lock1 is lock2

    def test_different_keys_different_locks(self):
        lock1 = get_file_operation_lock("s1", "/a")
        lock2 = get_file_operation_lock("s1", "/b")
        assert lock1 is not lock2

    def test_creates_lock_on_demand(self):
        assert active_lock_count() == 0
        lock = get_file_operation_lock("s1", "/file")
        assert active_lock_count() == 1
        del lock


class TestAcquireFileOperationLock:
    def setup_method(self):
        clear_locks()

    def test_acquire_release(self):
        with acquire_file_operation_lock("s1", "/file") as lock:
            assert lock.locked()

    def test_serializes_same_path(self):
        results = []

        def writer(value):
            with acquire_file_operation_lock("s1", "/file"):
                results.append(f"start-{value}")
                time.sleep(0.05)
                results.append(f"end-{value}")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each start should be followed by its end before next start
        for i in range(3):
            start_idx = results.index(f"start-{i}")
            assert results[start_idx + 1] == f"end-{i}"

    def test_different_paths_concurrent(self):
        results = []

        def writer(sandbox, path, value):
            with acquire_file_operation_lock(sandbox, path):
                results.append(f"start-{value}")
                time.sleep(0.05)
                results.append(f"end-{value}")

        t1 = threading.Thread(target=writer, args=("s1", "/a", 1))
        t2 = threading.Thread(target=writer, args=("s1", "/b", 2))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # Different paths should be able to run concurrently
        # Both starts should come before both ends (overlap)
        starts = [i for i, r in enumerate(results) if r.startswith("start")]
        ends = [i for i, r in enumerate(results) if r.startswith("end")]
        assert max(starts) < min(ends)

    def test_timeout_raises(self):
        lock = get_file_operation_lock("s1", "/file")
        lock.acquire()
        try:
            with pytest.raises(TimeoutError):
                with acquire_file_operation_lock("s1", "/file", timeout=0.1):
                    pass
        finally:
            lock.release()


class TestFileOperationLockManager:
    def test_creation(self):
        mgr = FileOperationLockManager()
        assert mgr.lock_count() == 0

    def test_get_lock(self):
        mgr = FileOperationLockManager()
        lock = mgr.get_lock("s1", "/file")
        assert lock is not None

    def test_same_key_same_lock(self):
        mgr = FileOperationLockManager()
        l1 = mgr.get_lock("s1", "/file")
        l2 = mgr.get_lock("s1", "/file")
        assert l1 is l2

    def test_acquire_context(self):
        mgr = FileOperationLockManager()
        with mgr.acquire("s1", "/file") as lock:
            assert lock.locked()

    def test_acquire_timeout(self):
        mgr = FileOperationLockManager()
        lock = mgr.get_lock("s1", "/file")
        lock.acquire()
        try:
            with pytest.raises(TimeoutError):
                with mgr.acquire("s1", "/file", timeout=0.1):
                    pass
        finally:
            lock.release()

    def test_clear(self):
        mgr = FileOperationLockManager()
        mgr.get_lock("s1", "/file")
        mgr.clear()
        assert mgr.lock_count() == 0

    def test_independent_from_module_level(self):
        mgr = FileOperationLockManager()
        clear_locks()
        module_lock = get_file_operation_lock("s1", "/file")
        mgr_lock = mgr.get_lock("s1", "/file")
        assert module_lock is not mgr_lock


class TestWeakRefCleanup:
    def setup_method(self):
        clear_locks()

    def test_lock_garbage_collected(self):
        import gc

        def create_and_drop():
            return get_file_operation_lock("s1", "/temp-file")

        lock = create_and_drop()
        assert active_lock_count() == 1
        del lock
        gc.collect()
        assert active_lock_count() == 0
