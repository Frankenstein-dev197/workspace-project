"""Tests for agent brain and planning system."""

import pytest

from daemon_engine.core.agent_brain import (
    AgentBrain,
    AgentPlanner,
    AgentStep,
    PlanItem,
    PlanItemStatus,
)


class TestPlanItem:
    def test_creation(self):
        item = PlanItem(text="Do something")
        assert item.text == "Do something"
        assert item.status == PlanItemStatus.PENDING

    def test_mark_done(self):
        item = PlanItem(text="Test")
        item.mark_done()
        assert item.status == PlanItemStatus.DONE
        assert item.completed_at is not None

    def test_mark_current(self):
        item = PlanItem(text="Test")
        item.mark_current()
        assert item.status == PlanItemStatus.CURRENT

    def test_mark_skipped(self):
        item = PlanItem(text="Test")
        item.mark_skipped()
        assert item.status == PlanItemStatus.SKIPPED

    def test_to_dict(self):
        item = PlanItem(text="Test")
        d = item.to_dict()
        assert d["text"] == "Test"
        assert d["status"] == "pending"


class TestAgentBrain:
    def test_creation(self):
        brain = AgentBrain(
            thinking="I need to read a file",
            evaluation_previous_goal="Successfully read file",
            memory="File contains Python code",
            next_goal="Edit the file",
        )
        assert brain.thinking == "I need to read a file"
        assert brain.next_goal == "Edit the file"

    def test_defaults(self):
        brain = AgentBrain()
        assert brain.thinking is None
        assert brain.next_goal is None

    def test_to_dict(self):
        brain = AgentBrain(thinking="Test", next_goal="Action")
        d = brain.to_dict()
        assert d["thinking"] == "Test"
        assert d["next_goal"] == "Action"


class TestAgentPlanner:
    def test_creation_empty(self):
        planner = AgentPlanner()
        assert len(planner.get_plan()) == 0
        assert planner.is_complete() is False

    def test_creation_with_initial_plan(self):
        planner = AgentPlanner(["Step 1", "Step 2", "Step 3"])
        assert len(planner.get_plan()) == 3
        assert planner.get_current_item() is not None
        assert planner.get_current_item().status == PlanItemStatus.CURRENT

    def test_add_plan_item(self):
        planner = AgentPlanner()
        planner.add_plan_item("New step")
        assert len(planner.get_plan()) == 1

    def test_add_plan_item_at_index(self):
        planner = AgentPlanner(["A", "C"])
        planner.add_plan_item("B", index=1)
        plan = planner.get_plan()
        assert plan[1].text == "B"
        assert plan[2].text == "C"

    def test_remove_plan_item(self):
        planner = AgentPlanner(["A", "B", "C"])
        assert planner.remove_plan_item(1) is True
        assert len(planner.get_plan()) == 2
        assert planner.get_plan()[1].text == "C"

    def test_remove_nonexistent(self):
        planner = AgentPlanner(["A"])
        assert planner.remove_plan_item(10) is False

    def test_advance(self):
        planner = AgentPlanner(["Step 1", "Step 2"])
        current = planner.get_current_item()
        assert current.text == "Step 1"
        next_item = planner.advance()
        assert next_item is not None
        assert next_item.text == "Step 2"
        assert planner.get_plan()[0].status == PlanItemStatus.DONE

    def test_advance_to_end(self):
        planner = AgentPlanner(["Only step"])
        planner.advance()
        assert planner.get_current_item() is None
        assert planner.is_complete() is True

    def test_skip_current(self):
        planner = AgentPlanner(["Step 1", "Step 2"])
        next_item = planner.skip_current()
        assert next_item is not None
        assert next_item.text == "Step 2"
        assert planner.get_plan()[0].status == PlanItemStatus.SKIPPED

    def test_update_plan(self):
        planner = AgentPlanner(["Step 1", "Step 2"])
        planner.advance()  # Complete step 1
        planner.update_plan(["New A", "New B", "New C"])
        plan = planner.get_plan()
        assert len(plan) == 3
        assert plan[0].text == "New A"

    def test_update_plan_preserves_completed(self):
        planner = AgentPlanner(["Step 1", "Step 2"])
        planner.advance()  # Complete step 1
        planner.update_plan(["Step 1", "Step 3"])
        plan = planner.get_plan()
        assert plan[0].status == PlanItemStatus.DONE

    def test_record_step(self):
        planner = AgentPlanner(["Step 1"])
        brain = AgentBrain(thinking="Doing step 1", next_goal="Complete it")
        step = planner.record_step(
            brain=brain,
            action="read_file",
            result="File contents",
        )
        assert step.step_number == 1
        assert step.action == "read_file"
        assert len(planner.get_steps()) == 1

    def test_record_multiple_steps(self):
        planner = AgentPlanner()
        for i in range(5):
            brain = AgentBrain(thinking=f"Step {i}")
            planner.record_step(brain=brain, action=f"action_{i}")
        steps = planner.get_steps()
        assert len(steps) == 5
        assert steps[-1].step_number == 5

    def test_get_plan_summary(self):
        planner = AgentPlanner(["A", "B", "C"])
        planner.advance()  # A done, B current
        summary = planner.get_plan_summary()
        assert summary["total_items"] == 3
        assert summary["done"] == 1
        assert summary["current"] == 1
        assert summary["completion_rate"] > 0

    def test_is_complete(self):
        planner = AgentPlanner(["A", "B"])
        planner.advance()
        planner.advance()
        assert planner.is_complete() is True

    def test_is_complete_with_skipped(self):
        planner = AgentPlanner(["A", "B"])
        planner.skip_current()
        planner.skip_current()
        assert planner.is_complete() is True

    def test_reset(self):
        planner = AgentPlanner(["A", "B"])
        planner.advance()
        planner.reset()
        plan = planner.get_plan()
        assert all(item.status == PlanItemStatus.PENDING for item in plan[1:])
        assert plan[0].status == PlanItemStatus.CURRENT
        assert len(planner.get_steps()) == 0

    def test_to_dict(self):
        planner = AgentPlanner(["A", "B"])
        brain = AgentBrain(thinking="Test")
        planner.record_step(brain=brain)
        d = planner.to_dict()
        assert "plan" in d
        assert "steps" in d
        assert "summary" in d
        assert len(d["plan"]) == 2
        assert len(d["steps"]) == 1


class TestAgentStep:
    def test_creation(self):
        brain = AgentBrain(thinking="Test")
        step = AgentStep(step_number=1, brain=brain, action="test")
        assert step.step_number == 1
        assert step.action == "test"
        assert step.success is True

    def test_failed_step(self):
        brain = AgentBrain(thinking="Test")
        step = AgentStep(
            step_number=1,
            brain=brain,
            action="test",
            success=False,
            error="Something went wrong",
        )
        assert step.success is False
        assert step.error == "Something went wrong"

    def test_to_dict(self):
        brain = AgentBrain(thinking="Test")
        step = AgentStep(step_number=1, brain=brain, action="test", result="ok")
        d = step.to_dict()
        assert d["step_number"] == 1
        assert d["action"] == "test"
        assert d["result"] == "ok"
        assert "brain" in d
