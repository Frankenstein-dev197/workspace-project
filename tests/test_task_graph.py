"""Tests for task graph system."""

import pytest
from pathlib import Path

from daemon_engine.core.task_graph import TaskGraph, GraphTask, TaskStatus


@pytest.fixture
def task_graph(tmp_path):
    """Create a fresh task graph in temp directory."""
    graph = TaskGraph(tasks_dir=tmp_path / "tasks")
    return graph


class TestGraphTask:
    def test_creation(self):
        task = GraphTask(id="t1", subject="Test task")
        assert task.id == "t1"
        assert task.status == TaskStatus.PENDING
        assert task.blocked_by == []

    def test_to_dict(self):
        task = GraphTask(id="t1", subject="Test", description="A test")
        d = task.to_dict()
        assert d["id"] == "t1"
        assert d["status"] == "pending"

    def test_from_dict(self):
        data = {"id": "t1", "subject": "Test", "status": "completed"}
        task = GraphTask.from_dict(data)
        assert task.status == TaskStatus.COMPLETED


class TestTaskGraph:
    def test_create_task(self, task_graph):
        task = task_graph.create_task("My task", "Description")
        assert task.id.startswith("task_")
        assert task.subject == "My task"
        assert task.status == TaskStatus.PENDING

    def test_get_task(self, task_graph):
        task = task_graph.create_task("Test")
        retrieved = task_graph.get_task(task.id)
        assert retrieved is not None
        assert retrieved.subject == "Test"

    def test_get_nonexistent(self, task_graph):
        assert task_graph.get_task("nonexistent") is None

    def test_list_tasks(self, task_graph):
        task_graph.create_task("Task 1")
        task_graph.create_task("Task 2")
        assert len(task_graph.list_tasks()) == 2

    def test_list_filtered_by_status(self, task_graph):
        t1 = task_graph.create_task("Task 1")
        task_graph.create_task("Task 2")
        task_graph.claim_task(t1.id, "agent1")
        pending = task_graph.list_tasks(status=TaskStatus.PENDING)
        assert len(pending) == 1
        in_progress = task_graph.list_tasks(status=TaskStatus.IN_PROGRESS)
        assert len(in_progress) == 1

    def test_list_filtered_by_owner(self, task_graph):
        t1 = task_graph.create_task("Task 1")
        t2 = task_graph.create_task("Task 2")
        task_graph.claim_task(t1.id, "agent1")
        task_graph.claim_task(t2.id, "agent2")
        agent1_tasks = task_graph.list_tasks(owner="agent1")
        assert len(agent1_tasks) == 1

    def test_can_start_no_deps(self, task_graph):
        task = task_graph.create_task("Test")
        assert task_graph.can_start(task.id) is True

    def test_can_start_with_completed_dep(self, task_graph):
        dep = task_graph.create_task("Dependency")
        task_graph.claim_task(dep.id, "agent1")
        task_graph.complete_task(dep.id)
        task = task_graph.create_task("Main", blocked_by=[dep.id])
        assert task_graph.can_start(task.id) is True

    def test_cannot_start_with_incomplete_dep(self, task_graph):
        dep = task_graph.create_task("Dependency")
        task = task_graph.create_task("Main", blocked_by=[dep.id])
        assert task_graph.can_start(task.id) is False

    def test_cannot_start_with_missing_dep(self, task_graph):
        task = task_graph.create_task("Main", blocked_by=["nonexistent"])
        assert task_graph.can_start(task.id) is False

    def test_claim_task(self, task_graph):
        task = task_graph.create_task("Test")
        assert task_graph.claim_task(task.id, "agent1") is True
        updated = task_graph.get_task(task.id)
        assert updated.owner == "agent1"
        assert updated.status == TaskStatus.IN_PROGRESS

    def test_claim_already_claimed(self, task_graph):
        task = task_graph.create_task("Test")
        task_graph.claim_task(task.id, "agent1")
        assert task_graph.claim_task(task.id, "agent2") is False

    def test_claim_blocked_task(self, task_graph):
        dep = task_graph.create_task("Dep")
        task = task_graph.create_task("Main", blocked_by=[dep.id])
        assert task_graph.claim_task(task.id, "agent1") is False

    def test_complete_task(self, task_graph):
        task = task_graph.create_task("Test")
        task_graph.claim_task(task.id, "agent1")
        assert task_graph.complete_task(task.id, "Done!") is True
        updated = task_graph.get_task(task.id)
        assert updated.status == TaskStatus.COMPLETED
        assert updated.result == "Done!"
        assert updated.completed_at is not None

    def test_complete_unclaimed(self, task_graph):
        task = task_graph.create_task("Test")
        assert task_graph.complete_task(task.id) is False

    def test_complete_unblocks_downstream(self, task_graph):
        dep = task_graph.create_task("Dependency")
        task = task_graph.create_task("Main", blocked_by=[dep.id])
        task_graph.claim_task(dep.id, "agent1")
        task_graph.complete_task(dep.id)
        assert task_graph.can_start(task.id) is True

    def test_fail_task(self, task_graph):
        task = task_graph.create_task("Test")
        assert task_graph.fail_task(task.id, "Something went wrong") is True
        updated = task_graph.get_task(task.id)
        assert updated.status == TaskStatus.FAILED
        assert "wrong" in updated.result

    def test_cancel_task(self, task_graph):
        task = task_graph.create_task("Test")
        assert task_graph.cancel_task(task.id) is True
        updated = task_graph.get_task(task.id)
        assert updated.status == TaskStatus.CANCELLED

    def test_get_ready_tasks(self, task_graph):
        t1 = task_graph.create_task("Task 1")
        task_graph.create_task("Task 2", blocked_by=[t1.id])
        ready = task_graph.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].id == t1.id

    def test_get_blocked_tasks(self, task_graph):
        t1 = task_graph.create_task("Task 1")
        t2 = task_graph.create_task("Task 2", blocked_by=[t1.id])
        blocked = task_graph.get_blocked_tasks()
        assert len(blocked) == 1
        assert blocked[0].id == t2.id

    def test_add_subtask(self, task_graph):
        parent = task_graph.create_task("Parent")
        child = task_graph.create_task("Child")
        assert task_graph.add_subtask(parent.id, child.id) is True
        updated = task_graph.get_task(parent.id)
        assert child.id in updated.subtasks

    def test_get_dependencies(self, task_graph):
        dep1 = task_graph.create_task("Dep 1")
        dep2 = task_graph.create_task("Dep 2")
        task = task_graph.create_task("Main", blocked_by=[dep1.id, dep2.id])
        deps = task_graph.get_dependencies(task.id)
        assert len(deps) == 2

    def test_get_dependents(self, task_graph):
        dep = task_graph.create_task("Dep")
        task_graph.create_task("Task 1", blocked_by=[dep.id])
        task_graph.create_task("Task 2", blocked_by=[dep.id])
        dependents = task_graph.get_dependents(dep.id)
        assert len(dependents) == 2

    def test_priority_ordering(self, task_graph):
        task_graph.create_task("Low", priority=1)
        task_graph.create_task("High", priority=10)
        task_graph.create_task("Med", priority=5)
        tasks = task_graph.list_tasks()
        assert tasks[0].subject == "High"
        assert tasks[-1].subject == "Low"

    def test_persistence(self, tmp_path):
        graph1 = TaskGraph(tasks_dir=tmp_path / "tasks")
        task = graph1.create_task("Persistent task")
        graph2 = TaskGraph(tasks_dir=tmp_path / "tasks")
        assert graph2.get_task(task.id) is not None

    def test_stats(self, task_graph):
        t1 = task_graph.create_task("Task 1")
        t2 = task_graph.create_task("Task 2", blocked_by=[t1.id])
        task_graph.claim_task(t1.id, "agent")
        task_graph.complete_task(t1.id)
        stats = task_graph.stats()
        assert stats["total_tasks"] == 2
        assert stats["by_status"]["completed"] == 1
        assert stats["ready_tasks"] >= 1

    def test_clear(self, task_graph):
        task_graph.create_task("Task 1")
        task_graph.create_task("Task 2")
        task_graph.clear()
        assert len(task_graph.list_tasks()) == 0

    def test_critical_path(self, task_graph):
        t1 = task_graph.create_task("Task 1")
        t2 = task_graph.create_task("Task 2", blocked_by=[t1.id])
        t3 = task_graph.create_task("Task 3", blocked_by=[t2.id])
        path = task_graph.get_critical_path()
        assert len(path) >= 3

    def test_to_dict_export(self, task_graph):
        task_graph.create_task("Task 1")
        task_graph.create_task("Task 2")
        export = task_graph.to_dict()
        assert len(export) == 2
        assert "id" in export[0]
