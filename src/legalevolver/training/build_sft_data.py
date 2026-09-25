from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def decompose_trajectory(trajectory: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Create first-plan, next-action, and final-judgment supervision."""
    case = trajectory["case"]
    steps = trajectory["steps"]
    if steps:
        yield {"task": "first_step_planning", "input": {"case": case}, "target": steps[0]["action"]}
    for index, step in enumerate(steps[1:], start=1):
        yield {
            "task": "next_action_prediction",
            "input": {"case": case, "history": steps[:index]},
            "target": step["action"],
        }
    yield {
        "task": "final_legal_judgment",
        "input": {"case": case, "history": steps},
        "target": trajectory["final_answer"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            for example in decompose_trajectory(json.loads(line)):
                target.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

