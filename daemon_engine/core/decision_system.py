"""Decision System: evaluates options and selects optimal actions.

Integrates decision-making patterns from Ruflo (learning-based routing) and
DeepSeek-Reasonix (task contracts with ablation). Provides utility-based,
rule-based, and learning-based decision strategies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from daemon_engine.models.base import BaseLLM, get_default_llm

logger = logging.getLogger(__name__)


class DecisionStrategy(Enum):
    UTILITY_BASED = "utility_based"
    RULE_BASED = "rule_based"
    LLM_BASED = "llm_based"
    LEARNING_BASED = "learning_based"


@dataclass
class Option:
    id: str
    label: str
    description: str = ""
    utility: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class Decision:
    selected_option: Option | None
    strategy: DecisionStrategy
    reasoning: str = ""
    alternatives: list[Option] = field(default_factory=list)
    confidence: float = 0.0


class DecisionSystem:
    """Evaluates options and selects the best action based on configured strategy."""

    LLM_DECISION_PROMPT = (
        "You are a decision-making agent. Given the following context and options, "
        "select the best option and explain your reasoning.\n\n"
        "Context: {context}\n\nOptions:\n{options}\n\n"
        "Respond with: SELECTED: <option_id>\nREASONING: <your explanation>"
    )

    def __init__(
        self,
        llm: BaseLLM | None = None,
        strategy: DecisionStrategy = DecisionStrategy.UTILITY_BASED,
        rules: list[Callable[[list[Option], dict], Option | None]] | None = None,
    ) -> None:
        self.llm = llm or get_default_llm()
        self.strategy = strategy
        self._rules = rules or []
        self._history: list[Decision] = []
        self._utility_weights: dict[str, float] = {}

    def add_rule(self, rule: Callable[[list[Option], dict], Option | None]) -> None:
        self._rules.append(rule)

    def set_utility_weight(self, factor: str, weight: float) -> None:
        self._utility_weights[factor] = weight

    def decide(
        self,
        options: list[Option],
        context: dict[str, Any] | None = None,
        strategy: DecisionStrategy | None = None,
    ) -> Decision:
        strat = strategy or self.strategy
        logger.info("Making decision with %s strategy, %d options", strat.value, len(options))
        if not options:
            return Decision(selected_option=None, strategy=strat, reasoning="No options available")
        if strat == DecisionStrategy.UTILITY_BASED:
            return self._utility_based_decide(options, context or {})
        elif strat == DecisionStrategy.RULE_BASED:
            return self._rule_based_decide(options, context or {})
        elif strat == DecisionStrategy.LLM_BASED:
            return self._llm_based_decide(options, context or {})
        elif strat == DecisionStrategy.LEARNING_BASED:
            return self._learning_based_decide(options, context or {})
        return self._utility_based_decide(options, context or {})

    def _utility_based_decide(self, options: list[Option], context: dict) -> Decision:
        for option in options:
            option.score = self._compute_utility(option, context)
        ranked = sorted(options, key=lambda o: o.score, reverse=True)
        best = ranked[0]
        reasoning = f"Selected {best.label} with utility score {best.score:.2f}"
        decision = Decision(
            selected_option=best,
            strategy=DecisionStrategy.UTILITY_BASED,
            reasoning=reasoning,
            alternatives=ranked[1:],
            confidence=min(best.score / 10.0, 1.0),
        )
        self._history.append(decision)
        return decision

    def _compute_utility(self, option: Option, context: dict) -> float:
        score = option.utility
        for factor, weight in self._utility_weights.items():
            if factor in option.metadata:
                score += option.metadata[factor] * weight
            elif factor in context:
                score += context[factor] * weight * 0.5
        for key, val in option.metadata.items():
            if isinstance(val, (int, float)) and key not in self._utility_weights:
                score += val * 0.1
        return score

    def _rule_based_decide(self, options: list[Option], context: dict) -> Decision:
        for rule in self._rules:
            selected = rule(options, context)
            if selected:
                decision = Decision(
                    selected_option=selected,
                    strategy=DecisionStrategy.RULE_BASED,
                    reasoning=f"Rule {rule.__name__} matched: {selected.label}",
                    alternatives=[o for o in options if o.id != selected.id],
                    confidence=0.9,
                )
                self._history.append(decision)
                return decision
        return self._utility_based_decide(options, context)

    def _llm_based_decide(self, options: list[Option], context: dict) -> Decision:
        options_text = "\n".join(
            f"  - ID: {o.id}, Label: {o.label}, Description: {o.description}" for o in options
        )
        context_text = ", ".join(f"{k}: {v}" for k, v in context.items()) if context else "N/A"
        prompt = self.LLM_DECISION_PROMPT.format(context=context_text, options=options_text)
        try:
            response = self.llm.chat([{"role": "user", "content": prompt}])
            selected_id, reasoning = self._parse_llm_response(response, options)
            selected = next((o for o in options if o.id == selected_id), options[0])
        except Exception as exc:
            logger.error("LLM decision failed: %s", exc)
            selected = options[0]
            reasoning = f"Fallback to first option due to LLM error: {exc}"
        decision = Decision(
            selected_option=selected,
            strategy=DecisionStrategy.LLM_BASED,
            reasoning=reasoning,
            alternatives=[o for o in options if o.id != selected.id],
            confidence=0.75,
        )
        self._history.append(decision)
        return decision

    def _parse_llm_response(self, response: str, options: list[Option]) -> tuple[str, str]:
        selected_id = ""
        reasoning = ""
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("SELECTED:"):
                selected_id = line[len("SELECTED:"):].strip()
            elif line.startswith("REASONING:"):
                reasoning = line[len("REASONING:"):].strip()
        if not selected_id and options:
            selected_id = options[0].id
        if not reasoning:
            reasoning = response[:200]
        return selected_id, reasoning

    def _learning_based_decide(self, options: list[Option], context: dict) -> Decision:
        scores: dict[str, float] = {}
        for option in options:
            past = sum(
                1 for d in self._history if d.selected_option and d.selected_option.id == option.id
            )
            success = sum(
                1
                for d in self._history
                if d.selected_option and d.selected_option.id == option.id and d.confidence > 0.5
            )
            success_rate = success / past if past > 0 else 0.5
            option.score = option.utility * 0.4 + success_rate * 5.0
            scores[option.id] = option.score
        ranked = sorted(options, key=lambda o: o.score, reverse=True)
        best = ranked[0]
        reasoning = (
            f"Selected {best.label} based on learning history "
            f"(score={best.score:.2f}, past selections={sum(1 for d in self._history if d.selected_option and d.selected_option.id == best.id)})"
        )
        decision = Decision(
            selected_option=best,
            strategy=DecisionStrategy.LEARNING_BASED,
            reasoning=reasoning,
            alternatives=ranked[1:],
            confidence=0.7,
        )
        self._history.append(decision)
        return decision

    def get_history(self) -> list[Decision]:
        return self._history

    def clear_history(self) -> None:
        self._history.clear()
