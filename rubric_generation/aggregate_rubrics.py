from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def canonical(text: str) -> str:
    return re.sub(r"\W+", "", text).lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge and deduplicate rubric candidates by tool and focus.")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    with args.candidates.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            for rubric in row.get("rubrics", []):
                key = (rubric["target_tool"], rubric["focus"])
                signature = canonical(rubric["criterion"])
                if signature and signature not in seen[key]:
                    seen[key].add(signature)
                    grouped[key].append(rubric)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for (tool, focus), rubrics in sorted(grouped.items()):
            handle.write(json.dumps({"target_tool": tool, "focus": focus, "rubrics": rubrics}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
