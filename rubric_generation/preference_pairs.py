from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def is_successful(row: dict[str, Any]) -> bool:
    return bool(row.get("success", False)) or float(row.get("outcome_score", 0.0)) >= 1.0


def build_pairs(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Contrast successful and failed trajectories for the same case."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case_id = str(row.get("case_id") or row.get("caseID") or "")
        if case_id:
            grouped[case_id].append(row)

    for case_id, candidates in grouped.items():
        preferred = [row for row in candidates if is_successful(row)]
        rejected = [row for row in candidates if not is_successful(row)]
        if not preferred or not rejected:
            continue
        best = max(preferred, key=lambda row: float(row.get("outcome_score", 1.0)))
        for bad in rejected:
            yield {
                "case_id": case_id,
                "case_fact": best.get("case_fact") or bad.get("case_fact") or "",
                "preferred": best,
                "rejected": bad,
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for pair in build_pairs(read_jsonl(args.trajectories)):
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
