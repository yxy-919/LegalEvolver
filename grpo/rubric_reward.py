from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.judge import JudgeClient


SUPPORTED_TOOLS = {
    "statute_search",
    "case_search",
    "guideline_search",
    "judicial_interpretation_search",
}


@dataclass(frozen=True)
class QueryEvent:
    tool: str
    query: str
    context: str


def extract_query_events(trajectory: dict[str, Any], context_turns: int = 8) -> list[QueryEvent]:
    """Extract every retrieval query with its preceding interaction context."""
    events: list[QueryEvent] = []
    turns = trajectory.get("turns", [])
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        call = turn.get("tool_call")
        if not isinstance(call, dict):
            continue
        tool = call.get("tool_name") or call.get("name")
        arguments = call.get("arguments", {})
        query = call.get("query") or (arguments.get("query") if isinstance(arguments, dict) else None)
        if tool not in SUPPORTED_TOOLS or not isinstance(query, str) or not query.strip():
            continue
        context = json.dumps(turns[max(0, index - context_turns):index], ensure_ascii=False)
        events.append(QueryEvent(str(tool), query.strip(), context))
    return events


class RubricBank:
    """Load JSONL rubrics organized by retrieval tool and reasoning focus."""

    def __init__(self, path: str | Path, max_per_group: int = 8) -> None:
        self.groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                tool, focus = row["target_tool"], row["focus"]
                self.groups.setdefault(tool, {})[focus] = list(row.get("rubrics", []))[:max_per_group]

    def for_tool(self, tool: str) -> dict[str, list[dict[str, Any]]]:
        if tool not in self.groups:
            raise KeyError(f"no rubrics found for tool: {tool}")
        return self.groups[tool]


class RubricReward:
    """Score each query with the tool- and focus-specific rubric selected by a judge."""

    def __init__(self, bank: RubricBank, judge: JudgeClient | None = None) -> None:
        self.bank = bank
        self.judge = judge or JudgeClient()

    def score_query(self, event: QueryEvent) -> float:
        groups = self.bank.for_tool(event.tool)
        result = self.judge.complete_json(
            "Evaluate one retrieval query in a legal judgment trajectory. Infer its reasoning "
            "focus from the interaction context, choose exactly one focus from the supplied "
            "rubric groups, and apply only that group. Return JSON with `focus`, "
            "`criterion_scores`, and `score`; score is in [0, 3].\n\n"
            f"TOOL: {event.tool}\n"
            f"CONTEXT: {event.context}\n"
            f"QUERY: {event.query}\n"
            f"RUBRIC GROUPS: {json.dumps(groups, ensure_ascii=False)}"
        )
        focus = str(result.get("focus", ""))
        if focus not in groups:
            raise ValueError(f"judge selected unknown focus: {focus}")
        return max(0.0, min(3.0, float(result["score"]))) / 3.0

    def score_trajectory(self, trajectory: dict[str, Any]) -> float:
        events = extract_query_events(trajectory)
        if not events:
            return 0.0
        return sum(self.score_query(event) for event in events) / len(events)
