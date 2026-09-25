from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .judge import JudgeClient


@dataclass(frozen=True)
class QueryEvent:
    tool: str
    query: str
    context: str


def extract_query_events(trajectory: dict[str, Any], context_turns: int = 8) -> list[QueryEvent]:
    """Extract tool, query, and preceding context for every retrieval call."""
    events: list[QueryEvent] = []
    turns = trajectory.get("turns", [])
    for index, turn in enumerate(turns):
        tool_call = turn.get("tool_call") if isinstance(turn, dict) else None
        if not isinstance(tool_call, dict):
            continue
        arguments = tool_call.get("arguments", {})
        query = arguments.get("query") if isinstance(arguments, dict) else None
        tool = tool_call.get("name")
        if not isinstance(tool, str) or not isinstance(query, str) or not query.strip():
            continue
        context = json.dumps(
            turns[max(0, index - context_turns) : index],
            ensure_ascii=False,
        )
        events.append(QueryEvent(tool=tool, query=query, context=context))
    return events


class RubricBank:
    """Rubrics grouped by retrieval tool and reasoning focus."""

    def __init__(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        self.groups: dict[str, dict[str, list[str]]] = payload["tools"]

    def for_tool(self, tool: str) -> dict[str, list[str]]:
        if tool not in self.groups:
            raise KeyError(f"no rubric groups found for tool: {tool}")
        return self.groups[tool]


class RubricReward:
    """Scores every query with the rubric selected for its tool and focus."""

    def __init__(self, rubric_bank: RubricBank, judge: JudgeClient | None = None) -> None:
        self.bank = rubric_bank
        self.judge = judge or JudgeClient()

    def score_query(self, event: QueryEvent) -> float:
        groups = self.bank.for_tool(event.tool)
        prompt = (
            "Evaluate one retrieval query from a legal judgment prediction trajectory.\n"
            "First infer the query's reasoning focus from the interaction context. "
            "Choose exactly one focus from the supplied rubric groups. Then apply only "
            "that group's criteria to score the query. Return JSON with keys focus, "
            "criterion_scores, and score. The final score must be between 0 and 1.\n\n"
            f"TOOL:\n{event.tool}\n\n"
            f"INTERACTION_CONTEXT:\n{event.context}\n\n"
            f"QUERY:\n{event.query}\n\n"
            f"RUBRIC_GROUPS:\n{json.dumps(groups, ensure_ascii=False)}"
        )
        result = self.judge.score(prompt)
        focus = str(result.get("focus", ""))
        if focus not in groups:
            raise ValueError(f"judge selected an unknown focus: {focus}")
        score = float(result["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError("rubric score must be in [0, 1]")
        return score

    def score_events(self, events: Iterable[QueryEvent]) -> float:
        scores = [self.score_query(event) for event in events]
        return sum(scores) / len(scores) if scores else 0.0

    def score_trajectory(self, trajectory: dict[str, Any]) -> float:
        return self.score_events(extract_query_events(trajectory))
