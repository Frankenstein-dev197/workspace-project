"""Tests for the decision system."""

import pytest

from daemon_engine.core.decision_system import DecisionSystem, DecisionStrategy, Option
from daemon_engine.models.providers import MockProvider


@pytest.fixture
def system():
    return DecisionSystem(llm=MockProvider())


@pytest.fixture
def options():
    return [
        Option(id="opt_1", label="Option A", description="First option", utility=7.0),
        Option(id="opt_2", label="Option B", description="Second option", utility=5.0),
        Option(id="opt_3", label="Option C", description="Third option", utility=3.0),
    ]


class TestDecisionSystem:
    def test_utility_based(self, system, options):
        decision = system.decide(options, strategy=DecisionStrategy.UTILITY_BASED)
        assert decision.selected_option.id == "opt_1"
        assert decision.confidence > 0

    def test_rule_based_no_match(self, system, options):
        system.add_rule(lambda opts, ctx: None)  # No-op rule
        decision = system.decide(options, strategy=DecisionStrategy.RULE_BASED)
        assert decision.selected_option is not None

    def test_rule_based_match(self, system, options):
        def always_first(opts, ctx):
            return opts[0] if opts else None

        system.add_rule(always_first)
        decision = system.decide(options, strategy=DecisionStrategy.RULE_BASED)
        assert decision.selected_option.id == "opt_1"

    def test_llm_based(self, system, options):
        decision = system.decide(options, strategy=DecisionStrategy.LLM_BASED)
        assert decision.selected_option is not None
        assert decision.reasoning

    def test_learning_based(self, system, options):
        decision1 = system.decide(options, strategy=DecisionStrategy.LEARNING_BASED)
        assert decision1.selected_option is not None
        decision2 = system.decide(options, strategy=DecisionStrategy.LEARNING_BASED)
        assert decision2.selected_option is not None

    def test_empty_options(self, system):
        decision = system.decide([], strategy=DecisionStrategy.UTILITY_BASED)
        assert decision.selected_option is None

    def test_history_tracking(self, system, options):
        system.decide(options)
        system.decide(options)
        assert len(system.get_history()) == 2

    def test_set_utility_weight(self, system, options):
        system.set_utility_weight("speed", 2.0)
        options[0].metadata["speed"] = 1.0
        options[1].metadata["speed"] = 10.0
        decision = system.decide(options, strategy=DecisionStrategy.UTILITY_BASED)
        assert decision.selected_option.id == "opt_2"

    def test_clear_history(self, system, options):
        system.decide(options)
        system.clear_history()
        assert len(system.get_history()) == 0
