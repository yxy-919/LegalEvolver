from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.judge import JudgeClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out revalidation of rubric criteria.")
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--heldout_pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum_gap", type=float, default=0.25)
    args = parser.parse_args()
    pairs = [json.loads(line) for line in args.heldout_pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    judge = JudgeClient()
    retained = []
    for line in args.rubrics.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        group = json.loads(line)
        kept = []
        for rubric in group.get("rubrics", []):
            result = judge.complete_json(
                "Score how well this criterion distinguishes preferred from rejected queries "
                "on the held-out pairs. Return JSON with `preferred_mean` and `rejected_mean`.\n"
                f"CRITERION: {json.dumps(rubric, ensure_ascii=False)}\n"
                f"HELD-OUT PAIRS: {json.dumps(pairs, ensure_ascii=False)}"
            )
            gap = float(result.get("preferred_mean", 0)) - float(result.get("rejected_mean", 0))
            if gap >= args.minimum_gap:
                kept.append({**rubric, "validation_gap": gap})
        if kept:
            retained.append({**group, "rubrics": kept})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retained), encoding="utf-8")


if __name__ == "__main__":
    main()
