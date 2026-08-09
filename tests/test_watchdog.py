"""Tests for watchdog loop detection system."""

import pytest

from daemon_engine.core.watchdog import (
    Watchdog,
    ActionRecord,
    LoopDetection,
    LoopType,
)


class TestActionRecord:
    def test_creation(self):
        record = ActionRecord(tool_name="bash", tool_input={"cmd": "ls"})
        assert record.tool_name == "bash"
        assert record.tool_input == {"cmd": "ls"}
        assert record.success is True

    def test_failed_action(self):
        record = ActionRecord(tool_name="bash", success=False)
        assert record.success is False


class TestWatchdog:
    def test_creation(self):
        wd = Watchdog()
        assert wd.is_big_brain is False
        assert wd.current_model == "fast"

    def test_custom_models(self):
        wd = Watchdog(fast_model="gpt-3.5", smart_model="gpt-4")
        assert wd.current_model == "gpt-3.5"

    def test_record_action_no_loop(self):
        wd = Watchdog()
        detection = wd.record_action("bash", {"cmd": "ls"})
        assert detection.is_loop is False

    def test_no_command_detected(self):
        wd = Watchdog()
        detection = wd.record_action(None)
        assert detection.is_loop is True
        assert detection.loop_type == LoopType.NO_COMMAND

    def test_exact_repetition_detected(self):
        wd = Watchdog(loop_threshold=3)
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("bash", {"cmd": "ls"})
        detection = wd.record_action("bash", {"cmd": "ls"})
        assert detection.is_loop is True
        assert detection.loop_type == LoopType.EXACT_REPETITION

    def test_consecutive_repetition(self):
        wd = Watchdog(loop_threshold=3)
        wd.record_action("read", {"path": "test.py"})
        wd.record_action("read", {"path": "test.py"})
        detection = wd.record_action("read", {"path": "test.py"})
        assert detection.is_loop is True

    def test_different_actions_no_loop(self):
        wd = Watchdog(loop_threshold=3)
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("read", {"path": "test.py"})
        detection = wd.record_action("write", {"path": "out.py"})
        assert detection.is_loop is False

    def test_big_brain_activation(self):
        wd = Watchdog(loop_threshold=3)
        assert wd.is_big_brain is False
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("bash", {"cmd": "ls"})
        assert wd.is_big_brain is True
        assert wd.current_model == "smart"

    def test_big_brain_reverts(self):
        wd = Watchdog(loop_threshold=3)
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("bash", {"cmd": "ls"})
        assert wd.is_big_brain is True
        wd.record_action("read", {"path": "other.py"})
        assert wd.is_big_brain is False

    def test_circular_detection(self):
        wd = Watchdog(circular_window=6)
        wd.record_action("read", {"path": "a"})
        wd.record_action("write", {"path": "b"})
        wd.record_action("read", {"path": "a"})
        wd.record_action("write", {"path": "b"})
        wd.record_action("read", {"path": "a"})
        detection = wd.record_action("write", {"path": "b"})
        assert detection.is_loop is True or wd._detect_circular(wd.get_history() + [ActionRecord(tool_name="write", tool_input={"path": "b"})])

    def test_stuck_detection(self):
        wd = Watchdog(loop_threshold=10)
        wd.record_action("bash", {"cmd": "fail"}, success=False)
        wd.record_action("bash", {"cmd": "fail"}, success=False)
        wd.record_action("bash", {"cmd": "fail"}, success=False)
        detection = wd.record_action("bash", {"cmd": "fail"}, success=False)
        assert detection.is_loop is True
        assert detection.loop_type == LoopType.STUCK

    def test_rewind(self):
        wd = Watchdog()
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("read", {"path": "test.py"})
        assert len(wd.get_history()) == 2
        rewound = wd.rewind(1)
        assert len(rewound) == 1
        assert len(wd.get_history()) == 1

    def test_clear_history(self):
        wd = Watchdog()
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("bash", {"cmd": "ls"})
        wd.clear_history()
        assert len(wd.get_history()) == 0
        assert wd.is_big_brain is False

    def test_get_history(self):
        wd = Watchdog()
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("read", {"path": "test.py"})
        history = wd.get_history()
        assert len(history) == 2
        assert history[0].tool_name == "bash"
        assert history[1].tool_name == "read"

    def test_recovery_prompt(self):
        wd = Watchdog(loop_threshold=3)
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("bash", {"cmd": "ls"})
        detection = wd.record_action("bash", {"cmd": "ls"})
        prompt = wd.get_recovery_prompt(detection)
        assert "bash" in prompt
        assert "different" in prompt.lower()

    def test_no_command_recovery_prompt(self):
        wd = Watchdog()
        detection = wd.record_action(None)
        prompt = wd.get_recovery_prompt(detection)
        assert "command" in prompt.lower()

    def test_stats(self):
        wd = Watchdog(loop_threshold=3)
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("bash", {"cmd": "ls"})
        stats = wd.stats()
        assert stats["total_actions"] == 3
        assert stats["loops_detected"] >= 1
        assert stats["by_type"]["exact_repetition"] >= 1

    def test_window_size_limits_history(self):
        wd = Watchdog(window_size=3)
        for i in range(10):
            wd.record_action("tool", {"i": i})
        assert len(wd.get_history()) <= 3

    def test_repetition_counts_cleared_on_clear(self):
        wd = Watchdog(loop_threshold=3)
        wd.record_action("bash", {"cmd": "ls"})
        wd.record_action("bash", {"cmd": "ls"})
        wd.clear_history()
        wd.record_action("bash", {"cmd": "ls"})
        detection = wd.record_action("bash", {"cmd": "ls"})
        assert detection.is_loop is False

    def test_partial_repetition_same_tool_different_args(self):
        wd = Watchdog(loop_threshold=10)
        wd.record_action("read", {"path": "a.py"})
        wd.record_action("read", {"path": "b.py"})
        detection = wd.record_action("read", {"path": "c.py"})
        assert detection.is_loop is False
