from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from grpo.rubric_reward import RubricBank, RubricReward, extract_query_events


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if not isinstance(value, list):
        value = [value]
    return {str(item).strip() for item in value if str(item).strip()}


def set_f1(gold: Any, prediction: Any) -> float:
    gold_set, pred_set = _as_set(gold), _as_set(prediction)
    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0
    overlap = len(gold_set & pred_set)
    precision, recall = overlap / len(pred_set), overlap / len(gold_set)
    return 2 * precision * recall / (precision + recall)


@dataclass(frozen=True)
class RewardWeights:
    outcome: float = 0.891
    format: float = 0.099
    rubric: float = 0.01


class LegalEvolverReward:
    """Combined outcome, format, and rubric reward described in the paper."""

    def __init__(self, rubric_reward: RubricReward, weights: RewardWeights | None = None) -> None:
        self.rubric_reward = rubric_reward
        self.weights = weights or RewardWeights()

    @staticmethod
    def outcome_score(prediction: dict[str, Any], reference: dict[str, Any]) -> float:
        law = set_f1(reference.get("relevant_articles"), prediction.get("relevant_articles"))
        charge = set_f1(reference.get("accusation"), prediction.get("accusation"))
        term = float(prediction.get("sentence_interval") == reference.get("sentence_interval"))
        return (law + charge + term) / 3.0

    @staticmethod
    def format_score(trajectory: dict[str, Any], prediction: Any) -> float:
        events = extract_query_events(trajectory)
        valid_tools = bool(events) and all(event.tool and event.query for event in events)
        try:
            answer = json.loads(prediction) if isinstance(prediction, str) else prediction
            valid_output = isinstance(answer, dict) and bool(answer)
        except json.JSONDecodeError:
            valid_output = False
        return 0.5 * float(valid_tools) + 0.5 * float(valid_output)

    def __call__(
        self,
        trajectory: dict[str, Any],
        prediction: dict[str, Any] | str,
        reference: dict[str, Any],
    ) -> dict[str, float]:
        try:
            parsed = json.loads(prediction) if isinstance(prediction, str) else prediction
        except json.JSONDecodeError:
            parsed = {}
        outcome = self.outcome_score(parsed, reference) if isinstance(parsed, dict) else 0.0
        formatting = self.format_score(trajectory, prediction)
        rubric = self.rubric_reward.score_trajectory(trajectory)
        total = (
            self.weights.outcome * outcome
            + self.weights.format * formatting
            + self.weights.rubric * rubric
        )
        return {"reward": total, "outcome": outcome, "format": formatting, "rubric": rubric}


_REWARD: LegalEvolverReward | None = None


def get_reward() -> LegalEvolverReward:
    """Create the reward lazily inside each training worker."""
    global _REWARD
    if _REWARD is None:
        weights = RewardWeights(
            outcome=float(os.getenv("OUTCOME_REWARD_WEIGHT", "0.891")),
            format=float(os.getenv("FORMAT_REWARD_WEIGHT", "0.099")),
            rubric=float(os.getenv("RUBRIC_REWARD_WEIGHT", "0.01")),
        )
        bank_path = Path(os.getenv("RUBRIC_BANK_PATH", "rubric_generation/final_rubric_bank.jsonl"))
        _REWARD = LegalEvolverReward(RubricReward(RubricBank(bank_path)), weights)
    return _REWARD


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """verl-compatible reward entry point."""
    del data_source
    extra_info = extra_info or {}
    try:
        prediction = json.loads(solution_str)
    except json.JSONDecodeError:
        prediction = extra_info.get("prediction", {})
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            ground_truth = {}
    trajectory = extra_info.get("trajectory") or {"turns": extra_info.get("turns", [])}
    result = get_reward()(trajectory, prediction, ground_truth if isinstance(ground_truth, dict) else {})
    return float(result["reward"])
