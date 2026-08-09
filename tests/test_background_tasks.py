"""Tests for background tasks."""

import time

import pytest

from daemon_engine.core.background_tasks import (
    BackgroundTaskManager,
    BackgroundTask,
    TaskStatus,
)


class TestBackgroundTask:
    def test_creation(self):
        task = BackgroundTask(id="bg_001", command="npm install")
        assert task.id == "bg_001"
        assert task.status == TaskStatus.PENDING

    def test_is_running(self):
        task = BackgroundTask(id="bg_001", command="test", status=TaskStatus.RUNNING)
        assert task.is_running is True

    def test_is_completed(self):
        task = BackgroundTask(id="bg_001", command="test", status=TaskStatus.COMPLETED)
        assert task.is_completed is True
        assert task.is_ready is True

    def test_is_failed(self):
        task = BackgroundTask(id="bg_001", command="test", status=TaskStatus.FAILED)
        assert task.is_failed is True
        assert task.is_ready is True

    def test_is_ready_for_failed(self):
        task = BackgroundTask(id="bg_001", command="test", status=TaskStatus.FAILED)
        assert task.is_ready is True

    def test_duration_not_started(self):
        task = BackgroundTask(id="bg_001", command="test")
        assert task.duration == 0.0

    def test_duration_running(self):
        task = BackgroundTask(id="bg_001", command="test", status=TaskStatus.RUNNING)
        task.started_at = time.time() - 1.0
        assert task.duration > 0.5

    def test_duration_completed(self):
        task = BackgroundTask(id="bg_001", command="test", status=TaskStatus.COMPLETED)
        task.started_at = 100.0
        task.completed_at = 105.0
        assert task.duration == 5.0

    def test_to_dict(self):
        task = BackgroundTask(id="bg_001", command="test", result="done")
        d = task.to_dict()
        assert d["id"] == "bg_001"
        assert d["command"] == "test"
        assert d["result"] == "done"


class TestBackgroundTaskManager:
    def test_creation(self):
        manager = BackgroundTaskManager()
        assert len(manager.list_tasks()) == 0

    def test_dispatch_simple(self):
        manager = BackgroundTaskManager()

        def quick_task():
            return "done"

        bg_id = manager.dispatch(func=quick_task)
        assert bg_id.startswith("bg_")
        manager.wait_for(bg_id, timeout=5)
        task = manager.get_task(bg_id)
        assert task is not None
        assert task.is_completed
        assert task.result == "done"

    def test_dispatch_with_input(self):
        manager = BackgroundTaskManager()

        def add(a, b):
            return str(a + b)

        bg_id = manager.dispatch(func=add, tool_input={"a": 2, "b": 3})
        manager.wait_for(bg_id, timeout=5)
        task = manager.get_task(bg_id)
        assert task.result == "5"

    def test_dispatch_failure(self):
        manager = BackgroundTaskManager()

        def failing_task():
            raise ValueError("Task error")

        bg_id = manager.dispatch(func=failing_task)
        manager.wait_for(bg_id, timeout=5)
        task = manager.get_task(bg_id)
        assert task.is_failed
        assert "Task error" in task.error

    def test_is_slow_operation(self):
        manager = BackgroundTaskManager()
        assert manager.is_slow_operation("bash", {"command": "npm install"}) is True
        assert manager.is_slow_operation("bash", {"command": "ls"}) is False
        assert manager.is_slow_operation("read", {"command": "npm install"}) is False

    def test_should_run_background_explicit(self):
        manager = BackgroundTaskManager()
        assert manager.should_run_background("bash", {"command": "ls", "run_in_background": True}) is True

    def test_should_run_background_heuristic(self):
        manager = BackgroundTaskManager()
        assert manager.should_run_background("bash", {"command": "npm install"}) is True
        assert manager.should_run_background("bash", {"command": "ls"}) is False

    def test_get_status(self):
        manager = BackgroundTaskManager()
        bg_id = manager.dispatch(func=lambda: "done")
        manager.wait_for(bg_id, timeout=5)
        status = manager.get_status(bg_id)
        assert status == TaskStatus.COMPLETED

    def test_get_status_nonexistent(self):
        manager = BackgroundTaskManager()
        assert manager.get_status("nonexistent") is None

    def test_collect_results(self):
        manager = BackgroundTaskManager()
        bg_id = manager.dispatch(func=lambda: "result data")
        manager.wait_for(bg_id, timeout=5)
        results = manager.collect_results()
        assert len(results) == 1
        assert results[0]["task_id"] == bg_id
        assert results[0]["status"] == "completed"
        assert results[0]["result"] == "result data"

    def test_collect_results_removes_tasks(self):
        manager = BackgroundTaskManager()
        manager.dispatch(func=lambda: "done")
        manager.wait_all(timeout=5)
        manager.collect_results()
        assert len(manager.list_tasks()) == 0

    def test_collect_results_only_ready(self):
        manager = BackgroundTaskManager()
        manager.dispatch(func=lambda: time.sleep(10))
        results = manager.collect_results()
        assert len(results) == 0

    def test_has_ready(self):
        manager = BackgroundTaskManager()
        bg_id = manager.dispatch(func=lambda: "done")
        manager.wait_for(bg_id, timeout=5)
        assert manager.has_ready() is True

    def test_has_running(self):
        manager = BackgroundTaskManager()
        manager.dispatch(func=lambda: time.sleep(10))
        assert manager.has_running() is True

    def test_cancel(self):
        manager = BackgroundTaskManager()
        bg_id = manager.dispatch(func=lambda: time.sleep(10))
        assert manager.cancel(bg_id) is True
        task = manager.get_task(bg_id)
        assert task.status == TaskStatus.CANCELLED

    def test_cancel_nonexistent(self):
        manager = BackgroundTaskManager()
        assert manager.cancel("nonexistent") is False

    def test_list_tasks(self):
        manager = BackgroundTaskManager()
        manager.dispatch(func=lambda: time.sleep(10))
        manager.dispatch(func=lambda: time.sleep(10))
        assert len(manager.list_tasks()) == 2

    def test_wait_all(self):
        manager = BackgroundTaskManager()
        manager.dispatch(func=lambda: "done1")
        manager.dispatch(func=lambda: "done2")
        manager.wait_all(timeout=5)
        assert manager.has_running() is False

    def test_stats(self):
        manager = BackgroundTaskManager()
        bg_id = manager.dispatch(func=lambda: "done")
        manager.wait_for(bg_id, timeout=5)
        stats = manager.stats()
        assert stats["total_dispatched"] == 1
        assert stats["total_completed"] == 1
        assert stats["ready"] >= 1

    def test_clear(self):
        manager = BackgroundTaskManager()
        bg_id = manager.dispatch(func=lambda: "done")
        manager.wait_for(bg_id, timeout=5)
        manager.clear()
        assert len(manager.list_tasks()) == 0

    def test_dispatch_tool(self):
        manager = BackgroundTaskManager()

        def bash_handler(command=""):
            return f"output: {command}"

        bg_id = manager.dispatch_tool(
            handler=bash_handler,
            tool_name="bash",
            tool_input={"command": "ls"},
        )
        manager.wait_for(bg_id, timeout=5)
        task = manager.get_task(bg_id)
        assert task.tool_name == "bash"
        assert "ls" in task.result

    def test_concurrent_dispatch(self):
        manager = BackgroundTaskManager()
        ids = []
        for i in range(5):
            ids.append(manager.dispatch(func=lambda x=i: f"result_{x}"))
        manager.wait_all(timeout=10)
        assert len(ids) == 5
        results = manager.collect_results()
        assert len(results) == 5

    def test_dispatch_preserves_command(self):
        manager = BackgroundTaskManager()
        bg_id = manager.dispatch(
            func=lambda: "done",
            command="npm install",
        )
        task = manager.get_task(bg_id)
        assert task.command == "npm install"
