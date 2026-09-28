from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.judge import JudgeClient


PROMPT = """Compare the preferred and rejected trajectories for the same legal case.
Identify how query formulation contributed to their different outcomes. Abstract the
case-level difference into reusable query criteria. Organize each criterion by the
retrieval tool and reasoning focus (legal_qualification or sentencing). Do not retain
case identifiers or case-specific conclusions. Return JSON with a `rubrics` list;
each item must contain target_tool, focus, criterion, and a 0--3 scoring_guide.

CASE FACT:
{case_fact}

PREFERRED TRAJECTORY:
{preferred}

REJECTED TRAJECTORY:
{rejected}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    judge = JudgeClient()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.pairs.open(encoding="utf-8") as source, args.output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            pair = json.loads(line)
            result = judge.complete_json(
                PROMPT.format(
                    case_fact=pair.get("case_fact", ""),
                    preferred=json.dumps(pair["preferred"], ensure_ascii=False),
                    rejected=json.dumps(pair["rejected"], ensure_ascii=False),
                )
            )
            target.write(json.dumps({"case_id": pair["case_id"], **result}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
