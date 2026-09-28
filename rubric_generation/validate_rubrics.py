from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from utils.judge import JudgeClient


FOCUS_ALIASES = {
    "legal_qualification": "legal_qualification",
    "sentencing": "sentencing",
    "full_repair": "full_repair",
    # The released bank uses this legacy name for the paper's Full Repair focus.
    "general_rubric": "full_repair",
}


@dataclass(frozen=True)
class QueryComparison:
    pair_id: str
    pair_type: str
    case_fact: str
    preferred_queries: list[dict[str, Any]]
    rejected_queries: list[dict[str, Any]]

    def as_prompt_record(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "pair_type": self.pair_type,
            "case_fact": self.case_fact,
            "preferred_queries": self.preferred_queries,
            "rejected_queries": self.rejected_queries,
        }


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def canonical_focus(value: Any) -> str:
    return FOCUS_ALIASES.get(str(value or "").strip(), "")


def tool_name(call: dict[str, Any]) -> str:
    return str(call.get("tool_name") or call.get("name") or call.get("tool") or "")


def query_text(call: dict[str, Any]) -> str:
    arguments = call.get("arguments")
    query = call.get("query")
    if query is None and isinstance(arguments, dict):
        query = arguments.get("query")
    return str(query or "").strip()


def extract_tool_queries(trajectory: Any, target_tool: str) -> list[dict[str, Any]]:
    """Extract queries to one tool together with their local interaction context."""
    if not isinstance(trajectory, dict):
        return []

    extracted: list[dict[str, Any]] = []
    calls = trajectory.get("tool_calls")
    if isinstance(calls, list):
        for index, call in enumerate(calls):
            if not isinstance(call, dict) or tool_name(call) != target_tool:
                continue
            query = query_text(call)
            if query:
                extracted.append(
                    {
                        "call_index": call.get("call_index", index),
                        "query": query,
                        "context": call.get("context") or call.get("reasoning") or "",
                    }
                )
        return extracted

    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        return []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        call = turn.get("tool_call")
        if not isinstance(call, dict) or tool_name(call) != target_tool:
            continue
        query = query_text(call)
        if query:
            extracted.append(
                {
                    "call_index": index,
                    "query": query,
                    "context": turns[max(0, index - 8):index],
                }
            )
    return extracted


def trajectory_side(pair: dict[str, Any], side: str) -> Any:
    return pair.get(f"{side}_trajectory") or pair.get(side) or {}


def critical_query_comparison(
    pair: dict[str, Any], target_tool: str, pair_id: str
) -> QueryComparison | None:
    """Build a direct comparison at the query-rewriting intervention point."""
    critical = pair.get("critical_query_pair")
    if not isinstance(critical, dict):
        return None
    critical_tool = str(
        critical.get("tool_type")
        or critical.get("target_tool")
        or critical.get("tool_name")
        or ""
    )
    preferred_query = str(critical.get("preferred_query") or "").strip()
    rejected_query = str(critical.get("rejected_query") or "").strip()
    if critical_tool != target_tool or not preferred_query or not rejected_query:
        return None
    call_index = (
        critical.get("call_index")
        if "call_index" in critical
        else critical.get("target_call_index")
    )
    return QueryComparison(
        pair_id=pair_id,
        pair_type="rewrite_critical",
        case_fact=str(pair.get("case_fact") or ""),
        preferred_queries=[
            {
                "call_index": call_index,
                "query": preferred_query,
                "context": critical.get("preferred_context") or "",
            }
        ],
        rejected_queries=[
            {
                "call_index": call_index,
                "query": rejected_query,
                "context": critical.get("rejected_context") or "",
            }
        ],
    )


def natural_query_comparison(
    pair: dict[str, Any], target_tool: str, pair_id: str
) -> QueryComparison | None:
    """Compare only queries issued to the same tool in natural trajectory pairs."""
    preferred = extract_tool_queries(trajectory_side(pair, "preferred"), target_tool)
    rejected = extract_tool_queries(trajectory_side(pair, "rejected"), target_tool)
    if not preferred or not rejected:
        return None
    return QueryComparison(
        pair_id=pair_id,
        pair_type="natural",
        case_fact=str(pair.get("case_fact") or ""),
        preferred_queries=preferred,
        rejected_queries=rejected,
    )


def select_comparisons(
    pairs: list[dict[str, Any]], target_tool: str, focus: str
) -> list[QueryComparison]:
    """Select held-out evidence for exactly one tool-focus rubric group."""
    selected: list[QueryComparison] = []
    target_focus = canonical_focus(focus)
    for index, pair in enumerate(pairs):
        if canonical_focus(pair.get("focus")) != target_focus:
            continue
        pair_id = str(pair.get("pair_id") or f"heldout_{index}")
        critical = pair.get("critical_query_pair")
        if isinstance(critical, dict) and critical:
            # A rewriting pair is evidence only for the tool used at the
            # intervention point; do not fall back to unrelated trajectory calls.
            comparison = critical_query_comparison(pair, target_tool, pair_id)
        else:
            comparison = natural_query_comparison(pair, target_tool, pair_id)
        if comparison is not None:
            selected.append(comparison)
    return selected


def batches(items: list[QueryComparison], batch_size: int) -> Iterable[list[QueryComparison]]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def score_batch(
    judge: JudgeClient,
    rubric: dict[str, Any],
    comparisons: list[QueryComparison],
) -> dict[str, tuple[float, float]]:
    result = judge.complete_json(
        "Evaluate whether one query criterion distinguishes preferred from rejected "
        "queries in held-out legal-search preference pairs. For `rewrite_critical` "
        "pairs, score only the two queries at the rewritten intervention point. For "
        "`natural` pairs, score the two same-tool query sets using their local contexts. "
        "Do not use the final judgment outcome to assign scores. Apply the supplied "
        "0--3 scoring guide and return JSON as {`scores`: [{`pair_id`, "
        "`preferred_score`, `rejected_score`}]}.\n\n"
        f"CRITERION: {json.dumps(rubric, ensure_ascii=False)}\n"
        f"HELD-OUT COMPARISONS: {json.dumps([item.as_prompt_record() for item in comparisons], ensure_ascii=False)}"
    )
    requested_ids = {item.pair_id for item in comparisons}
    scores: dict[str, tuple[float, float]] = {}
    for row in result.get("scores", []):
        if not isinstance(row, dict):
            continue
        pair_id = str(row.get("pair_id") or "")
        if pair_id not in requested_ids:
            continue
        try:
            preferred = max(0.0, min(3.0, float(row["preferred_score"])))
            rejected = max(0.0, min(3.0, float(row["rejected_score"])))
        except (KeyError, TypeError, ValueError):
            continue
        scores[pair_id] = (preferred, rejected)
    return scores


def validate_criterion(
    judge: JudgeClient,
    rubric: dict[str, Any],
    comparisons: list[QueryComparison],
    batch_size: int,
) -> dict[str, float]:
    scores: dict[str, tuple[float, float]] = {}
    for batch in batches(comparisons, batch_size):
        scores.update(score_batch(judge, rubric, batch))
    if not scores:
        return {
            "examples": 0,
            "preferred_mean": 0.0,
            "rejected_mean": 0.0,
            "gap": 0.0,
            "win_rate": 0.0,
        }
    values = list(scores.values())
    preferred_mean = sum(preferred for preferred, _ in values) / len(values)
    rejected_mean = sum(rejected for _, rejected in values) / len(values)
    win_rate = sum(preferred > rejected for preferred, rejected in values) / len(values)
    return {
        "examples": len(values),
        "preferred_mean": preferred_mean,
        "rejected_mean": rejected_mean,
        "gap": preferred_mean - rejected_mean,
        "win_rate": win_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tool- and focus-specific held-out rubric validation.")
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--heldout_pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--minimum_examples", type=int, default=5)
    parser.add_argument("--minimum_gap", type=float, default=0.25)
    parser.add_argument("--minimum_win_rate", type=float, default=0.60)
    args = parser.parse_args()

    if args.batch_size < 1 or args.minimum_examples < 1:
        parser.error("batch_size and minimum_examples must be positive")

    pairs = list(read_jsonl(args.heldout_pairs))
    judge = JudgeClient()
    retained: list[dict[str, Any]] = []
    for group in read_jsonl(args.rubrics):
        target_tool = str(group.get("target_tool") or "")
        focus = str(group.get("focus") or "")
        comparisons = select_comparisons(pairs, target_tool, focus)
        kept = []
        for rubric in group.get("rubrics", []):
            metrics = validate_criterion(judge, rubric, comparisons, args.batch_size)
            if (
                metrics["examples"] >= args.minimum_examples
                and metrics["gap"] >= args.minimum_gap
                and metrics["win_rate"] >= args.minimum_win_rate
            ):
                kept.append(
                    {
                        **rubric,
                        "validation_examples": int(metrics["examples"]),
                        "validation_preferred_mean": metrics["preferred_mean"],
                        "validation_rejected_mean": metrics["rejected_mean"],
                        "validation_gap": metrics["gap"],
                        "validation_win_rate": metrics["win_rate"],
                    }
                )
        if kept:
            retained.append({**group, "rubric_count": len(kept), "rubrics": kept})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in retained:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
