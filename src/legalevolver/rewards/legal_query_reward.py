from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from .rubric_reward import RubricReward


@dataclass(frozen=True)
class RewardWeights:
    outcome: float = 0.891
    format: float = 0.099
    rubric: float = 0.01

    def __post_init__(self) -> None:
        if abs(self.outcome + self.format + self.rubric - 1.0) > 1e-8:
            raise ValueError("reward weights must sum to one")


class LegalEvolverReward:
    """Combines final-outcome, output-format, and intermediate-query rewards."""

    def __init__(
        self,
        rubric_reward: RubricReward,
        outcome_scorer: Callable[[Any, Any], float],
        weights: RewardWeights | None = None,
    ) -> None:
        self.rubric_reward = rubric_reward
        self.outcome_scorer = outcome_scorer
        self.weights = weights or RewardWeights()

    @staticmethod
    def format_score(tool_calls: Iterable[dict[str, Any]], final_answer: Any) -> float:
        calls = list(tool_calls)
        valid_calls = True
        for call in calls:
            if not isinstance(call, dict):
                valid_calls = False
                break
            tool = call.get("tool") or call.get("name")
            arguments = call.get("arguments", {})
            query = call.get("query") or (
                arguments.get("query") if isinstance(arguments, dict) else None
            )
            if not isinstance(tool, str) or not isinstance(query, str) or not query.strip():
                valid_calls = False
                break
        try:
            answer = json.loads(final_answer) if isinstance(final_answer, str) else final_answer
            valid_answer = isinstance(answer, dict) and bool(answer)
        except json.JSONDecodeError:
            valid_answer = False
        return 0.5 * float(valid_calls) + 0.5 * float(valid_answer)

    def __call__(
        self,
        trajectory: dict[str, Any],
        prediction: Any,
        reference: Any,
    ) -> dict[str, float]:
        calls = [
            turn["tool_call"]
            for turn in trajectory.get("turns", [])
            if isinstance(turn, dict) and isinstance(turn.get("tool_call"), dict)
        ]
        outcome = float(self.outcome_scorer(prediction, reference))
        formatting = self.format_score(calls, prediction)
        rubric = self.rubric_reward.score_trajectory(trajectory)
        total = (
            self.weights.outcome * outcome
            + self.weights.format * formatting
            + self.weights.rubric * rubric
        )
        return {"reward": total, "outcome": outcome, "format": formatting, "rubric": rubric}
