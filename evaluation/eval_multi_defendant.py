from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = str(value or "").split("</think>")[-1].strip()
    answers = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S)
    candidate = answers[-1] if answers else text
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.S)
    if fenced:
        candidate = fenced.group(1)
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(candidate[start:end + 1])
                return payload if isinstance(payload, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def normalize_articles(value: Any) -> set[str]:
    articles = set()
    for item in as_list(value):
        match = re.search(r"\d+", str(item))
        if match:
            articles.add(match.group(0))
    return articles


def normalize_charges(value: Any) -> set[str]:
    charges = set()
    for item in as_list(value):
        charge = str(item or "").strip().strip("[]")
        if charge.endswith("罪"):
            charge = charge[:-1]
        if charge:
            charges.add(charge)
    return charges


def imprisonment_to_interval(value: Any) -> int:
    if isinstance(value, dict):
        if value.get("death_penalty") or value.get("life_imprisonment"):
            return 0
        months = int(value.get("imprisonment") or 0)
    else:
        months = int(value or 0)
    boundaries = (240, 120, 84, 60, 36, 24, 12, 9, 6, 0)
    for label, boundary in enumerate(boundaries):
        if months > boundary:
            return label
    return 10


def normalize_sentence_interval(value: Any) -> int | None:
    if isinstance(value, int):
        return value if 0 <= value <= 10 else None
    match = re.search(r"\d+", str(value or ""))
    if not match:
        return None
    label = int(match.group(0))
    return label if 0 <= label <= 10 else None


def per_defendant_weight(count: int) -> float:
    return 1.0 if count <= 1 else math.log(count, 2) / count


def set_metrics(
    gold_sets: Sequence[set[str]],
    pred_sets: Sequence[set[str]],
    weights: Sequence[float],
) -> dict[str, float]:
    total = float(sum(weights))
    exact = precision = recall = f1 = 0.0
    for gold, pred, weight in zip(gold_sets, pred_sets, weights):
        overlap = len(gold & pred)
        item_precision = overlap / len(pred) if pred else 0.0
        item_recall = overlap / len(gold) if gold else 0.0
        item_f1 = (
            2 * item_precision * item_recall / (item_precision + item_recall)
            if item_precision + item_recall
            else 0.0
        )
        exact += weight * float(gold == pred)
        precision += weight * item_precision
        recall += weight * item_recall
        f1 += weight * item_f1
    return {
        "accuracy": exact / total if total else 0.0,
        "precision": precision / total if total else 0.0,
        "recall": recall / total if total else 0.0,
        "f1": f1 / total if total else 0.0,
    }


def classification_metrics(
    gold: Sequence[int],
    pred: Sequence[int],
    weights: Sequence[float],
) -> dict[str, float]:
    total = float(sum(weights))
    accuracy = sum(w for g, p, w in zip(gold, pred, weights) if g == p) / total if total else 0.0
    labels = sorted(set(gold) | set(pred))
    precisions, recalls, f1s = [], [], []
    for label in labels:
        tp = sum(w for g, p, w in zip(gold, pred, weights) if g == label and p == label)
        fp = sum(w for g, p, w in zip(gold, pred, weights) if g != label and p == label)
        fn = sum(w for g, p, w in zip(gold, pred, weights) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": accuracy,
        "precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }


def gold_records(row: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for outcome in row.get("outcomes") or []:
        articles, charges, penalties = set(), set(), []
        for judgment in outcome.get("judgment") or []:
            articles.update(normalize_articles(judgment.get("article")))
            charges.update(normalize_charges(judgment.get("standard_accusation") or judgment.get("accusation")))
            penalties.append(imprisonment_to_interval(judgment.get("penalty")))
        records.append({"name": str(outcome.get("name") or ""), "articles": articles, "charges": charges, "penalties": penalties})
    return records


def prediction_records(row: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    payload = extract_payload(row.get("predict") or row.get("prediction"))
    defendants = payload.get("defendants")
    if not isinstance(defendants, list):
        return [], bool(payload)
    records = []
    for defendant in defendants:
        if not isinstance(defendant, dict):
            continue
        articles, charges, penalties = set(), set(), []
        for judgment in defendant.get("judgments") or []:
            if not isinstance(judgment, dict):
                continue
            articles.update(normalize_articles(judgment.get("relevant_articles")))
            charges.update(normalize_charges(judgment.get("accusation")))
            penalties.append(normalize_sentence_interval(judgment.get("sentence_interval")))
        records.append({"name": str(defendant.get("name") or ""), "articles": articles, "charges": charges, "penalties": penalties})
    return records, True


def align_by_name(
    gold: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, int]:
    empty = {"name": "", "articles": set(), "charges": set(), "penalties": []}
    by_name = {row["name"]: row for row in predictions if row["name"]}
    used: set[int] = set()
    aligned, missing = [], 0
    for gold_row in gold:
        prediction = by_name.get(gold_row["name"])
        if prediction is None:
            missing += 1
            prediction = empty
        else:
            used.add(id(prediction))
        aligned.append(prediction)
    extra = sum(id(row) not in used for row in predictions)
    return aligned, missing, extra


def evaluate(gold_path: Path, pred_path: Path) -> dict[str, Any]:
    gold_by_id = {str(row.get("caseID") or row.get("case_id")): row for row in load_jsonl(gold_path)}
    article_gold, article_pred, charge_gold, charge_pred = [], [], [], []
    penalty_gold, penalty_pred, set_weights, penalty_weights = [], [], [], []
    summary = {"evaluated_cases": 0, "evaluated_defendants": 0, "parse_failed": 0, "missing_defendants": 0, "extra_defendants": 0}
    for row in load_jsonl(pred_path):
        case_id = str(row.get("caseID") or row.get("case_id"))
        if case_id not in gold_by_id:
            continue
        gold = gold_records(gold_by_id[case_id])
        predictions, parsed = prediction_records(row)
        if not parsed:
            summary["parse_failed"] += 1
        aligned, missing, extra = align_by_name(gold, predictions)
        summary["missing_defendants"] += missing
        summary["extra_defendants"] += extra
        summary["evaluated_cases"] += 1
        weight = per_defendant_weight(len(gold))
        for gold_row, pred_row in zip(gold, aligned):
            summary["evaluated_defendants"] += 1
            article_gold.append(gold_row["articles"])
            article_pred.append(pred_row["articles"])
            charge_gold.append(gold_row["charges"])
            charge_pred.append(pred_row["charges"])
            set_weights.append(weight)
            for index, gold_penalty in enumerate(gold_row["penalties"]):
                prediction = pred_row["penalties"][index] if index < len(pred_row["penalties"]) else None
                penalty_gold.append(gold_penalty)
                penalty_pred.append(prediction if prediction is not None else -1)
                penalty_weights.append(weight)
    return {
        "article": set_metrics(article_gold, article_pred, set_weights),
        "charge": set_metrics(charge_gold, charge_pred, set_weights),
        "penalty": classification_metrics(penalty_gold, penalty_pred, penalty_weights),
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multi-defendant legal judgments.")
    parser.add_argument("--gold_path", type=Path, required=True)
    parser.add_argument("--pred_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path)
    args = parser.parse_args()
    metrics = evaluate(args.gold_path, args.pred_path)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output_path:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
