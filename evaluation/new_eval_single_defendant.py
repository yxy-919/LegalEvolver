from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if "```" in text:
        text = text.replace("```json", "").replace("```", "").strip()
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(text[start:end + 1])
                return payload if isinstance(payload, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def normalize_label(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "__MISSING__").strip().replace("[", "").replace("]", "")


def accuracy(gold: list[str], pred: list[str]) -> float:
    return sum(a == b for a, b in zip(gold, pred)) / max(1, len(gold))


def macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / max(1, len(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate single-defendant legal judgments.")
    parser.add_argument("--gold_path", type=Path, required=True)
    parser.add_argument("--pred_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path)
    args = parser.parse_args()
    gold = {str(row.get("caseID") or row.get("case_id")): row for row in load_jsonl(args.gold_path)}
    fields = {
        "article": "relevant_articles",
        "charge": "accusation",
        "penalty": "sentence_interval",
    }
    values = {name: ([], []) for name in fields}
    parsed_rows = 0
    for row in load_jsonl(args.pred_path):
        case_id = str(row.get("caseID") or row.get("case_id"))
        if case_id not in gold:
            continue
        prediction = extract_payload(row.get("predict") or row.get("prediction"))
        if prediction:
            parsed_rows += 1
        reference = gold[case_id].get("label") or gold[case_id].get("meta") or gold[case_id]
        for name, field in fields.items():
            values[name][0].append(normalize_label(reference.get(field)))
            values[name][1].append(normalize_label(prediction.get(field)))
    metrics = {
        name: {"accuracy": accuracy(gold_values, pred_values), "macro_f1": macro_f1(gold_values, pred_values)}
        for name, (gold_values, pred_values) in values.items()
    }
    metrics["summary"] = {"evaluated": len(values["article"][0]), "parsed": parsed_rows}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
