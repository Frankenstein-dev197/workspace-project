"""File operation locks: per-sandbox-path locking for concurrent safety.

Integrates DeerFlow file_operation_lock pattern:
- WeakValueDictionary-backed lock registry: prevents memory leaks in
  long-running processes by automatically removing locks when no longer
  referenced by any thread.
- Per (sandbox_id, path) granularity: concurrent operations on different
  files in the same sandbox proceed without blocking; same file serializes.
- get_file_operation_lock: returns a shared Lock for a key, creating it
  on first access.
- context manager: acquire_lock for clean lock acquire/release.

Without this, concurrent file writes from multiple agent threads can
corrupt files or produce interleaved writes. The weak reference ensures
locks for abandoned paths don't accumulate indefinitely.
"""

from __future__ import annotations

import threading
import weakref
from contextlib import contextmanager
from typing import Any, Generator, Hashable

_LockKey = tuple[str, str]

_FILE_OPERATION_LOCKS: weakref.WeakValueDictionary[_LockKey, threading.Lock] = (
    weakref.WeakValueDictionary()
)
_FILE_OPERATION_LOCKS_GUARD = threading.Lock()


def get_file_operation_lock_key(
    sandbox_id: str,
    path: str,
) -> _LockKey:
    """Create a lock key from sandbox ID and file path."""
    if not sandbox_id:
        sandbox_id = "default"
    return (sandbox_id, path)


def get_file_operation_lock(
    sandbox_id: str,
    path: str,
) -> threading.Lock:
    """Get or create a Lock for a (sandbox_id, path) pair.

    Uses WeakValueDictionary so locks are garbage-collected when no
    thread holds a reference, preventing memory leaks in long-running
    processes.
    """
    lock_key = get_file_operation_lock_key(sandbox_id, path)
    with _FILE_OPERATION_LOCKS_GUARD:
        lock = _FILE_OPERATION_LOCKS.get(lock_key)
        if lock is None:
            lock = threading.Lock()
            _FILE_OPERATION_LOCKS[lock_key] = lock
        return lock


@contextmanager
def acquire_file_operation_lock(
    sandbox_id: str,
    path: str,
    timeout: float | None = None,
) -> Generator[threading.Lock, None, None]:
    """Context manager that acquires and releases a file operation lock.

    Raises TimeoutError if the lock cannot be acquired within timeout.
    """
    lock = get_file_operation_lock(sandbox_id, path)
    if timeout is None:
        lock.acquire()
        acquired = True
    else:
        acquired = lock.acquire(timeout=timeout)
    if not acquired:
        raise TimeoutError(
            f"Could not acquire file operation lock for "
            f"sandbox={sandbox_id}, path={path} within {timeout}s"
        )
    try:
        yield lock
    finally:
        lock.release()


def active_lock_count() -> int:
    """Return the number of currently tracked locks (for diagnostics)."""
    with _FILE_OPERATION_LOCKS_GUARD:
        return len(_FILE_OPERATION_LOCKS)


def clear_locks() -> None:
    """Clear all file operation locks (for testing)."""
    with _FILE_OPERATION_LOCKS_GUARD:
        _FILE_OPERATION_LOCKS.clear()


class FileOperationLockManager:
    """Object-oriented wrapper for file operation locks.

    Provides a scoped interface for managing per-sandbox file locks,
    useful when you want instance-scoped state rather than module-level.
    """

    def __init__(self) -> None:
        self._locks: weakref.WeakValueDictionary[_LockKey, threading.Lock] = (
            weakref.WeakValueDictionary()
        )
        self._guard = threading.Lock()

    def get_lock(self, sandbox_id: str, path: str) -> threading.Lock:
        """Get or create a lock for a (sandbox_id, path) pair."""
        key = get_file_operation_lock_key(sandbox_id, path)
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    @contextmanager
    def acquire(
        self,
        sandbox_id: str,
        path: str,
        timeout: float | None = None,
    ) -> Generator[threading.Lock, None, None]:
        """Acquire and release a lock as a context manager."""
        lock = self.get_lock(sandbox_id, path)
        if timeout is None:
            lock.acquire()
            acquired = True
        else:
            acquired = lock.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError(
                f"Could not acquire lock for sandbox={sandbox_id}, path={path}"
            )
        try:
            yield lock
        finally:
            lock.release()

    def lock_count(self) -> int:
        """Return the number of currently tracked locks."""
        with self._guard:
            return len(self._locks)

    def clear(self) -> None:
        """Clear all locks."""
        with self._guard:
            self._locks.clear()
