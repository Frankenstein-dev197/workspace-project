"""Tests for episodic action history."""

import pytest

from daemon_engine.core.action_history import (
    EpisodicActionHistory,
    Episode,
    AgentAction,
    ActionResult,
    ActionStatus,
)


class TestActionStatus:
    def test_values(self):
        assert ActionStatus.SUCCESS.value == "success"
        assert ActionStatus.ERROR.value == "error"
        assert ActionStatus.INTERRUPTED_BY_HUMAN.value == "interrupted_by_human"
        assert ActionStatus.DID_NOT_FINISH.value == "did_not_finish"


class TestAgentAction:
    def test_creation(self):
        action = AgentAction(use_tool="bash", args={"cmd": "ls"})
        assert action.use_tool == "bash"
        assert action.args == {"cmd": "ls"}

    def test_defaults(self):
        action = AgentAction(use_tool="search")
        assert action.args == {}
        assert action.reasoning == ""
        assert action.raw_message == ""

    def test_to_dict(self):
        action = AgentAction(
            use_tool="bash",
            args={"cmd": "ls"},
            reasoning="list files",
            raw_message="run ls",
        )
        d = action.to_dict()
        assert d["use_tool"] == "bash"
        assert d["args"] == {"cmd": "ls"}
        assert d["reasoning"] == "list files"


class TestActionResult:
    def test_creation(self):
        result = ActionResult(status=ActionStatus.SUCCESS, output="file1\nfile2")
        assert result.status == ActionStatus.SUCCESS
        assert result.output == "file1\nfile2"

    def test_defaults(self):
        result = ActionResult()
        assert result.status == ActionStatus.SUCCESS
        assert result.output == ""

    def test_to_dict(self):
        result = ActionResult(
            status=ActionStatus.ERROR,
            reason="command not found",
            error="bash: foo: command not found",
        )
        d = result.to_dict()
        assert d["status"] == "error"
        assert d["reason"] == "command not found"

    def test_str_with_output(self):
        result = ActionResult(output="done")
        assert str(result) == "done"

    def test_str_without_output(self):
        result = ActionResult(status=ActionStatus.ERROR)
        assert "[error]" in str(result)


class TestEpisode:
    def test_creation(self):
        action = AgentAction(use_tool="bash")
        episode = Episode(action=action)
        assert episode.action.use_tool == "bash"
        assert episode.result is None
        assert episode.summary is None

    def test_is_complete_without_result(self):
        episode = Episode(action=AgentAction(use_tool="bash"))
        assert episode.is_complete is False

    def test_is_complete_with_result(self):
        episode = Episode(
            action=AgentAction(use_tool="bash"),
            result=ActionResult(status=ActionStatus.SUCCESS),
        )
        assert episode.is_complete is True

    def test_format_success(self):
        episode = Episode(
            action=AgentAction(use_tool="bash", reasoning="list files"),
            result=ActionResult(status=ActionStatus.SUCCESS, output="file1"),
        )
        formatted = episode.format()
        assert "bash" in formatted
        assert "list files" in formatted
        assert "success" in formatted
        assert "file1" in formatted

    def test_format_error(self):
        episode = Episode(
            action=AgentAction(use_tool="bash"),
            result=ActionResult(
                status=ActionStatus.ERROR,
                reason="not found",
                error="err details",
            ),
        )
        formatted = episode.format()
        assert "error" in formatted
        assert "not found" in formatted
        assert "err details" in formatted

    def test_format_interrupted(self):
        episode = Episode(
            action=AgentAction(use_tool="bash"),
            result=ActionResult(
                status=ActionStatus.INTERRUPTED_BY_HUMAN,
                feedback="stop please",
            ),
        )
        formatted = episode.format()
        assert "interrupted" in formatted
        assert "stop please" in formatted

    def test_format_no_result(self):
        episode = Episode(action=AgentAction(use_tool="bash"))
        formatted = episode.format()
        assert "did_not_finish" in formatted

    def test_to_dict(self):
        episode = Episode(
            action=AgentAction(use_tool="bash"),
            result=ActionResult(status=ActionStatus.SUCCESS),
            summary="ran bash",
        )
        d = episode.to_dict()
        assert d["action"]["use_tool"] == "bash"
        assert d["result"]["status"] == "success"
        assert d["summary"] == "ran bash"


class TestEpisodicActionHistory:
    def test_creation(self):
        history = EpisodicActionHistory()
        assert len(history) == 0
        assert bool(history) is False

    def test_current_episode_empty(self):
        history = EpisodicActionHistory()
        assert history.current_episode is None

    def test_register_action(self):
        history = EpisodicActionHistory()
        action = AgentAction(use_tool="bash")
        history.register_action(action)
        assert len(history) == 1
        assert history.current_episode is not None
        assert history.current_episode.action.use_tool == "bash"

    def test_register_result(self):
        history = EpisodicActionHistory()
        history.register_action(AgentAction(use_tool="bash"))
        history.register_result(ActionResult(status=ActionStatus.SUCCESS))
        assert history.current_episode is None  # cursor moved past
        assert history.episodes[0].is_complete is True

    def test_register_result_without_action_raises(self):
        history = EpisodicActionHistory()
        with pytest.raises(RuntimeError):
            history.register_result(ActionResult())

    def test_register_result_twice_raises(self):
        history = EpisodicActionHistory()
        history.register_action(AgentAction(use_tool="bash"))
        history.register_result(ActionResult())
        history.cursor = 0  # hack to test
        with pytest.raises(ValueError):
            history.register_result(ActionResult())

    def test_register_action_twice_raises(self):
        history = EpisodicActionHistory()
        history.register_action(AgentAction(use_tool="bash"))
        with pytest.raises(ValueError):
            history.register_action(AgentAction(use_tool="search"))

    def test_multiple_episodes(self):
        history = EpisodicActionHistory()
        for i in range(3):
            history.register_action(AgentAction(use_tool=f"tool{i}"))
            history.register_result(ActionResult(status=ActionStatus.SUCCESS))
        assert len(history) == 3
        assert history.completed_count == 3

    def test_append_user_feedback(self):
        history = EpisodicActionHistory()
        history.append_user_feedback("good job")
        assert "good job" in history.pending_user_feedback

    def test_set_summary(self):
        history = EpisodicActionHistory()
        history.register_action(AgentAction(use_tool="bash"))
        history.register_result(ActionResult())
        history.set_summary(0, "ran bash command")
        assert history.episodes[0].summary == "ran bash command"

    def test_set_summary_invalid_index(self):
        history = EpisodicActionHistory()
        history.set_summary(99, "summary")  # should not raise

    def test_get_messages_empty(self):
        history = EpisodicActionHistory()
        assert history.get_messages() == []

    def test_get_messages_single_episode(self):
        history = EpisodicActionHistory()
        history.register_action(
            AgentAction(use_tool="bash", raw_message="run ls")
        )
        history.register_result(
            ActionResult(status=ActionStatus.SUCCESS, output="file1")
        )
        messages = history.get_messages()
        assert len(messages) >= 1

    def test_get_messages_with_summary(self):
        history = EpisodicActionHistory()
        # Create many episodes so some get summarized
        for i in range(10):
            history.register_action(
                AgentAction(use_tool=f"tool{i}", raw_message=f"action {i}")
            )
            history.register_result(
                ActionResult(status=ActionStatus.SUCCESS, output=f"result {i}")
            )
        # Set summaries for older episodes
        for i in range(6):
            history.set_summary(i, f"summary of step {i}")

        messages = history.get_messages(full_message_count=4, max_tokens=10000)
        assert len(messages) > 0
        # Should contain summary text for older episodes
        has_summary = any("Progress" in m["content"] for m in messages)
        assert has_summary

    def test_get_messages_max_tokens_cap(self):
        history = EpisodicActionHistory()
        for i in range(20):
            history.register_action(
                AgentAction(
                    use_tool=f"tool{i}",
                    raw_message=f"action {i} " * 100,
                )
            )
            history.register_result(
                ActionResult(status=ActionStatus.SUCCESS, output=f"result {i}" * 50)
            )
        messages = history.get_messages(
            full_message_count=2,
            max_tokens=100,
        )
        # Should be limited by token cap
        total_content = sum(len(m["content"]) for m in messages)
        # Rough check: 100 tokens ~ 400 chars
        assert total_content < 5000

    def test_get_messages_includes_feedback(self):
        history = EpisodicActionHistory()
        history.register_action(AgentAction(use_tool="bash"))
        history.register_result(ActionResult())
        history.append_user_feedback("nice work")
        messages = history.get_messages()
        has_feedback = any("User Feedback" in m["content"] for m in messages)
        assert has_feedback

    def test_clear(self):
        history = EpisodicActionHistory()
        history.register_action(AgentAction(use_tool="bash"))
        history.register_result(ActionResult())
        history.append_user_feedback("fb")
        history.clear()
        assert len(history) == 0
        assert history.pending_user_feedback == []

    def test_to_dict(self):
        history = EpisodicActionHistory()
        history.register_action(AgentAction(use_tool="bash"))
        history.register_result(ActionResult())
        d = history.to_dict()
        assert "episodes" in d
        assert len(d["episodes"]) == 1
        assert d["cursor"] == 1

    def test_completed_count(self):
        history = EpisodicActionHistory()
        history.register_action(AgentAction(use_tool="bash"))
        assert history.completed_count == 0
        history.register_result(ActionResult())
        assert history.completed_count == 1

    def test_getitem(self):
        history = EpisodicActionHistory()
        history.register_action(AgentAction(use_tool="bash"))
        episode = history[0]
        assert episode.action.use_tool == "bash"


class TestThreadSafety:
    def test_concurrent_register(self):
        import threading

        history = EpisodicActionHistory()
        errors = []

        def worker(i):
            try:
                for j in range(5):
                    history.register_action(
                        AgentAction(use_tool=f"tool-{i}-{j}")
                    )
                    history.register_result(ActionResult())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(history) == 25
