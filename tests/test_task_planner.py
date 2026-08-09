"""Tests for the task planner."""

import pytest

from daemon_engine.core.task_planner import Task, TaskPlanner, TaskPriority, TaskStatus
from daemon_engine.models.providers import MockProvider


@pytest.fixture
def planner():
    return TaskPlanner(llm=MockProvider())


class TestTask:
    def test_task_creation(self):
        task = Task(description="Test task")
        assert task.id
        assert task.description == "Test task"
        assert task.status == TaskStatus.PENDING

    def test_task_add_subtask(self):
        parent = Task(description="Parent")
        child = Task(description="Child")
        parent.add_subtask(child.id)
        assert child.id in parent.subtask_ids

    def test_task_is_leaf(self):
        task = Task(description="Leaf")
        assert task.is_leaf()
        task.add_subtask("sub-id")
        assert not task.is_leaf()


class TestTaskStatus:
    def test_is_terminal(self):
        assert TaskStatus.COMPLETED.is_terminal
        assert TaskStatus.FAILED.is_terminal
        assert not TaskStatus.PENDING.is_terminal
        assert not TaskStatus.IN_PROGRESS.is_terminal

    def test_is_active(self):
        assert TaskStatus.IN_PROGRESS.is_active
        assert TaskStatus.BLOCKED.is_active
        assert not TaskStatus.PENDING.is_active


class TestTaskPlanner:
    def test_create_task(self, planner):
        task = planner.create_task("Test task")
        assert task.description == "Test task"
        assert planner.get_task(task.id) is task

    def test_decompose(self, planner):
        root = planner.decompose("Build a web app")
        assert root.description == "Build a web app"
        assert len(root.subtask_ids) > 0

    def test_get_subtasks(self, planner):
        root = planner.decompose("Do something")
        subs = planner.get_subtasks(root.id)
        assert len(subs) > 0

    def test_get_ready_tasks(self, planner):
        task1 = planner.create_task("Task 1")
        task2 = planner.create_task("Task 2")
        ready = planner.get_ready_tasks()
        assert task1 in ready
        assert task2 in ready

    def test_update_status(self, planner):
        task = planner.create_task("Task")
        assert planner.update_status(task.id, TaskStatus.COMPLETED, "Done") is True
        assert task.status == TaskStatus.COMPLETED
        assert task.result == "Done"

    def test_assign_agent(self, planner):
        task = planner.create_task("Task")
        assert planner.assign_agent(task.id, "agent-123") is True
        assert task.assigned_agent_id == "agent-123"

    def test_get_task_tree(self, planner):
        root = planner.decompose("Complex goal")
        tree = planner.get_task_tree()
        assert "Complex goal" in tree

    def test_heuristic_decompose(self, planner):
        subs = planner._heuristic_decompose("Some goal")
        assert len(subs) == 4
