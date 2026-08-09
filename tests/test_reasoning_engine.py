"""Tests for the reasoning engine."""

import pytest

from daemon_engine.core.reasoning_engine import (
    ReasoningEngine,
    ReasoningStrategy,
    ReasoningResult,
    ReasoningStep,
)
from daemon_engine.models.providers import MockProvider


@pytest.fixture
def engine():
    return ReasoningEngine(llm=MockProvider())


class TestReasoningStrategies:
    def test_chain_of_thought(self, engine):
        result = engine.reason("How to solve X?", ReasoningStrategy.CHAIN_OF_THOUGHT)
        assert result.strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
        assert len(result.steps) > 0
        assert result.conclusion

    def test_react(self, engine):
        result = engine.reason("What action to take?", ReasoningStrategy.REACT)
        assert result.strategy == ReasoningStrategy.REACT
        assert len(result.steps) > 0

    def test_reflection(self, engine):
        result = engine.reason("Reflect on this problem", ReasoningStrategy.REFLECTION)
        assert result.strategy == ReasoningStrategy.REFLECTION
        assert result.conclusion

    def test_tree_of_thought(self, engine):
        result = engine.reason("Best approach for Y?", ReasoningStrategy.TREE_OF_THOUGHT)
        assert result.strategy == ReasoningStrategy.TREE_OF_THOUGHT
        assert len(result.steps) > 0

    def test_self_consistency(self, engine):
        result = engine.reason("Consistent answer for Z?", ReasoningStrategy.SELF_CONSISTENCY)
        assert result.strategy == ReasoningStrategy.SELF_CONSISTENCY
        assert len(result.alternatives) == 3

    def test_compare_strategies(self, engine):
        results = engine.compare_strategies("Complex problem")
        assert len(results) == len(ReasoningStrategy)
        for strategy in ReasoningStrategy:
            assert strategy.value in results


class TestReasoningResult:
    def test_num_steps(self):
        result = ReasoningResult(
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            steps=[ReasoningStep(step_number=1, thought="a"), ReasoningStep(step_number=2, thought="b")],
        )
        assert result.num_steps == 2
