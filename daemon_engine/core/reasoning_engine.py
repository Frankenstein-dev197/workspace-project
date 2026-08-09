"""Reasoning Engine: chain-of-thought and multi-step reasoning.

Integrates reasoning patterns from DeepSeek-Reasonix (structured reasoning
chains, ablation testing) and Transformers (model-based reasoning). Supports
ReAct, Chain-of-Thought, Tree-of-Thought, and Reflection strategies.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from daemon_engine.models.base import BaseLLM, get_default_llm

logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REACT = "react"
    TREE_OF_THOUGHT = "tree_of_thought"
    REFLECTION = "reflection"
    SELF_CONSISTENCY = "self_consistency"


@dataclass
class ReasoningStep:
    step_number: int
    thought: str
    evidence: str = ""
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class ReasoningResult:
    strategy: ReasoningStrategy
    steps: list[ReasoningStep] = field(default_factory=list)
    conclusion: str = ""
    confidence: float = 0.0
    alternatives: list[str] = field(default_factory=list)

    @property
    def num_steps(self) -> int:
        return len(self.steps)


class ReasoningEngine:
    """Multi-strategy reasoning engine for complex problem-solving."""

    COT_PROMPT = (
        "Think step by step to solve the following problem. "
        "For each step, state your reasoning clearly.\n\nProblem: {problem}"
    )
    REACT_PROMPT = (
        "Use the ReAct framework. For each step:\n"
        "Thought: [your reasoning]\n"
        "Action: [what you would do]\n"
        "Observation: [expected result]\n\nProblem: {problem}"
    )
    REFLECTION_PROMPT = (
        "First, propose a solution to this problem.\n"
        "Then, critically reflect on your solution — identify flaws, "
        "edge cases, and improvements.\n"
        "Finally, provide a refined solution.\n\nProblem: {problem}"
    )
    TOT_PROMPT = (
        "Generate 3 different approaches to solve this problem. "
        "Evaluate each approach, then select the best one and explain why.\n\nProblem: {problem}"
    )

    def __init__(self, llm: BaseLLM | None = None) -> None:
        self.llm = llm or get_default_llm()

    def reason(
        self,
        problem: str,
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT,
        context: str | None = None,
    ) -> ReasoningResult:
        logger.info("Reasoning with %s strategy", strategy.value)
        if strategy == ReasoningStrategy.CHAIN_OF_THOUGHT:
            return self._chain_of_thought(problem, context)
        elif strategy == ReasoningStrategy.REACT:
            return self._react(problem, context)
        elif strategy == ReasoningStrategy.REFLECTION:
            return self._reflection(problem, context)
        elif strategy == ReasoningStrategy.TREE_OF_THOUGHT:
            return self._tree_of_thought(problem, context)
        elif strategy == ReasoningStrategy.SELF_CONSISTENCY:
            return self._self_consistency(problem, context)
        return self._chain_of_thought(problem, context)

    def _chain_of_thought(self, problem: str, context: str | None) -> ReasoningResult:
        prompt = self.COT_PROMPT.format(problem=problem)
        if context:
            prompt += f"\n\nContext: {context}"
        response = self.llm.chat([{"role": "user", "content": prompt}])
        steps = self._parse_steps(response)
        return ReasoningResult(
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            steps=steps,
            conclusion=response,
            confidence=0.8,
        )

    def _react(self, problem: str, context: str | None) -> ReasoningResult:
        prompt = self.REACT_PROMPT.format(problem=problem)
        if context:
            prompt += f"\n\nContext: {context}"
        response = self.llm.chat([{"role": "user", "content": prompt}])
        steps = self._parse_react_steps(response)
        return ReasoningResult(
            strategy=ReasoningStrategy.REACT,
            steps=steps,
            conclusion=response,
            confidence=0.75,
        )

    def _reflection(self, problem: str, context: str | None) -> ReasoningResult:
        prompt = self.REFLECTION_PROMPT.format(problem=problem)
        if context:
            prompt += f"\n\nContext: {context}"
        response = self.llm.chat([{"role": "user", "content": prompt}])
        steps = self._parse_steps(response)
        refined = self._extract_refined_solution(response)
        return ReasoningResult(
            strategy=ReasoningStrategy.REFLECTION,
            steps=steps,
            conclusion=refined,
            confidence=0.85,
            alternatives=[response],
        )

    def _tree_of_thought(self, problem: str, context: str | None) -> ReasoningResult:
        prompt = self.TOT_PROMPT.format(problem=problem)
        if context:
            prompt += f"\n\nContext: {context}"
        response = self.llm.chat([{"role": "user", "content": prompt}])
        alternatives = self._extract_alternatives(response)
        steps = self._parse_steps(response)
        return ReasoningResult(
            strategy=ReasoningStrategy.TREE_OF_THOUGHT,
            steps=steps,
            conclusion=response,
            confidence=0.82,
            alternatives=alternatives,
        )

    def _self_consistency(self, problem: str, context: str | None) -> ReasoningResult:
        prompt = self.COT_PROMPT.format(problem=problem)
        if context:
            prompt += f"\n\nContext: {context}"
        responses: list[str] = []
        for _ in range(3):
            responses.append(self.llm.chat([{"role": "user", "content": prompt}]))
        all_steps: list[ReasoningStep] = []
        for i, resp in enumerate(responses):
            all_steps.extend(self._parse_steps(resp, offset=i))
        conclusion = self._aggregate_responses(responses)
        return ReasoningResult(
            strategy=ReasoningStrategy.SELF_CONSISTENCY,
            steps=all_steps,
            conclusion=conclusion,
            confidence=0.88,
            alternatives=responses,
        )

    def _parse_steps(self, response: str, offset: int = 0) -> list[ReasoningStep]:
        steps: list[ReasoningStep] = []
        for i, line in enumerate(response.split("\n"), 1):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("Step") or line.startswith("-")):
                clean = line.lstrip("0123456789.-) ").replace("Step", "").strip(": ").strip()
                if clean:
                    steps.append(ReasoningStep(step_number=i + offset, thought=clean))
        if not steps:
            steps.append(ReasoningStep(step_number=1, thought=response[:500]))
        return steps

    def _parse_react_steps(self, response: str) -> list[ReasoningStep]:
        steps: list[ReasoningStep] = []
        current_thought = ""
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("Thought:"):
                current_thought = line[len("Thought:"):].strip()
            elif line.startswith("Action:"):
                if current_thought:
                    steps.append(
                        ReasoningStep(step_number=len(steps) + 1, thought=current_thought, evidence=line)
                    )
                    current_thought = ""
            elif line.startswith("Observation:") and steps:
                steps[-1].evidence = line[len("Observation:"):].strip()
        if not steps:
            steps = self._parse_steps(response)
        return steps

    def _extract_alternatives(self, response: str) -> list[str]:
        alternatives: list[str] = []
        for marker in ["Approach 1", "Approach 2", "Approach 3", "Option 1", "Option 2", "Option 3"]:
            idx = response.find(marker)
            if idx >= 0:
                end = response.find("Approach", idx + 1)
                if end < 0:
                    end = response.find("Option", idx + 1)
                if end < 0:
                    end = len(response)
                alternatives.append(response[idx:end].strip())
        return alternatives

    def _extract_refined_solution(self, response: str) -> str:
        markers = ["Refined solution", "Final solution", "Improved solution", "Revised"]
        for marker in markers:
            idx = response.lower().find(marker.lower())
            if idx >= 0:
                return response[idx:].strip()
        return response

    def _aggregate_responses(self, responses: list[str]) -> str:
        return responses[0] if responses else ""

    def compare_strategies(self, problem: str) -> dict[str, ReasoningResult]:
        results: dict[str, ReasoningResult] = {}
        for strategy in ReasoningStrategy:
            results[strategy.value] = self.reason(problem, strategy)
        return results
