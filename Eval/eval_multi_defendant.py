from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

try:
    from .common import (
        ChargeNormalizer,
        DEFAULT_BGE_MODEL_PATH,
        DEFAULT_CMDL_C2I_PATH,
        DEFAULT_CMDL_CRIME_ENCODINGS_PATH,
        classification_metrics,
        extract_answer_payload,
        imprisonment_to_interval,
        load_jsonl,
        load_jsonl_by_case_id,
        maybe_write_metrics,
        normalize_articles,
        normalize_sentence_interval,
        per_defendant_weight,
        print_metrics,
        set_metrics,
    )
except ImportError:
    from common import (  # type: ignore
        ChargeNormalizer,
        DEFAULT_BGE_MODEL_PATH,
        DEFAULT_CMDL_C2I_PATH,
        DEFAULT_CMDL_CRIME_ENCODINGS_PATH,
        classification_metrics,
        extract_answer_payload,
        imprisonment_to_interval,
        load_jsonl,
        load_jsonl_by_case_id,
        maybe_write_metrics,
        normalize_articles,
        normalize_sentence_interval,
        per_defendant_weight,
        print_metrics,
        set_metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multi-defendant CMDL SFT outputs on article, charge, and sentence interval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gold_path",
        type=Path,
        required=True,
        help="Gold CMDL-style JSONL file.",
    )
    parser.add_argument(
        "--pred_path",
        type=Path,
        required=True,
        help="Generated JSONL file from Eval.inference.run_agentic.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Optional metrics JSON output path.",
    )
    parser.add_argument(
        "--include_summary",
        action="store_true",
        help="Include parsing/alignment counts in addition to task metrics.",
    )
    parser.add_argument(
        "--c2i_path",
        type=Path,
        default=DEFAULT_CMDL_C2I_PATH,
        help="Charge vocabulary path used by BGE fallback.",
    )
    parser.add_argument(
        "--crime_encodings_path",
        type=Path,
        default=DEFAULT_CMDL_CRIME_ENCODINGS_PATH,
        help="Precomputed charge embeddings path used by BGE fallback.",
    )
    parser.add_argument(
        "--bge_model_path",
        default=DEFAULT_BGE_MODEL_PATH,
        help="BGE model path used by charge fallback matching.",
    )
    return parser.parse_args()


def gold_defendant_records(
    row: Dict[str, Any],
    charge_normalizer: ChargeNormalizer,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for outcome in row.get("outcomes") or []:
        article_set: set[str] = set()
        charge_set: set[str] = set()
        penalty_labels: List[int] = []
        for judgment in outcome.get("judgment") or []:
            article_set.update(normalize_articles(judgment.get("article"), keep_clause=False))
            charge_set.update(
                charge_normalizer.normalize_gold_many(
                    judgment.get("standard_accusation") or judgment.get("accusation")
                )
            )
            penalty_labels.append(imprisonment_to_interval(judgment.get("penalty") or {}))
        records.append(
            {
                "name": str(outcome.get("name") or ""),
                "articles": article_set,
                "charges": charge_set,
                "penalties": penalty_labels,
            }
        )
    return records


def pred_defendant_records(
    row: Dict[str, Any],
    charge_normalizer: ChargeNormalizer,
) -> tuple[List[Dict[str, Any]], bool]:
    payload = extract_answer_payload(str(row.get("predict") or "")) or {}
    records: List[Dict[str, Any]] = []
    defendants = payload.get("defendants")
    if not isinstance(defendants, list):
        return records, bool(payload)
    for defendant in defendants:
        if not isinstance(defendant, dict):
            continue
        name = str(defendant.get("name") or "")
        article_set: set[str] = set()
        charge_set: set[str] = set()
        penalty_labels: List[int | None] = []
        judgments = defendant.get("judgments")
        if not isinstance(judgments, list):
            judgments = []
        for judgment in judgments:
            if not isinstance(judgment, dict):
                continue
            article_set.update(normalize_articles(judgment.get("relevant_articles"), keep_clause=False))
            charge_set.update(charge_normalizer.normalize_pred_many(judgment.get("accusation")))
            penalty_labels.append(normalize_sentence_interval(judgment.get("sentence_interval")))
        records.append(
            {
                "name": name,
                "articles": article_set,
                "charges": charge_set,
                "penalties": penalty_labels,
            }
        )
    return records, bool(payload)


def align_predictions(
    gold_records: List[Dict[str, Any]],
    pred_records: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int, int]:
    empty = {"name": "", "articles": set(), "charges": set(), "penalties": []}
    if not gold_records:
        return [], 0, max(0, len(pred_records))

    extra_count = max(0, len(pred_records) - len(gold_records))

    by_name = {record["name"]: record for record in pred_records if record.get("name")}
    used_ids: set[int] = set()
    aligned: List[Dict[str, Any]] = []
    missing_count = 0

    for idx, gold in enumerate(gold_records):
        pred = by_name.get(gold["name"])
        if pred is not None:
            used_ids.add(id(pred))
        else:
            missing_count += 1
            pred = empty
        aligned.append(pred)

    unmatched_predictions = sum(1 for record in pred_records if id(record) not in used_ids)
    extra_count = max(extra_count, unmatched_predictions)
    return aligned, missing_count, extra_count


def evaluate(
    gold_path: Path,
    pred_path: Path,
    *,
    charge_normalizer: ChargeNormalizer,
    include_summary: bool = False,
) -> Dict[str, Any]:
    gold_by_id = load_jsonl_by_case_id(gold_path)
    pred_rows = load_jsonl(pred_path)

    article_gold_sets: List[set[str]] = []
    article_pred_sets: List[set[str]] = []
    charge_gold_sets: List[set[str]] = []
    charge_pred_sets: List[set[str]] = []
    penalty_true: List[int] = []
    penalty_pred: List[int] = []
    penalty_weights: List[float] = []
    set_weights: List[float] = []

    missing_gold = 0
    parse_failed = 0
    missing_defendant = 0
    extra_defendant = 0
    bad_penalty = 0
    evaluated_cases = 0
    evaluated_defendants = 0

    for pred_row in pred_rows:
        case_id = str(pred_row.get("caseID"))
        gold_row = gold_by_id.get(case_id)
        if gold_row is None:
            missing_gold += 1
            continue

        gold_records = gold_defendant_records(gold_row, charge_normalizer)
        pred_records, parsed = pred_defendant_records(pred_row, charge_normalizer)
        if not parsed:
            parse_failed += 1

        defendant_count = len(gold_records)
        weight = per_defendant_weight(defendant_count)
        aligned_preds, missing_count, extra_count = align_predictions(
            gold_records,
            pred_records,
        )
        missing_defendant += missing_count
        extra_defendant += extra_count
        evaluated_cases += 1

        for gold, pred in zip(gold_records, aligned_preds):
            evaluated_defendants += 1
            article_gold_sets.append(gold["articles"])
            article_pred_sets.append(pred["articles"])
            charge_gold_sets.append(gold["charges"])
            charge_pred_sets.append(pred["charges"])
            set_weights.append(weight)

            pred_penalties = pred["penalties"]
            for idx, gold_penalty in enumerate(gold["penalties"]):
                penalty_true.append(gold_penalty)
                pred_penalty = pred_penalties[idx] if idx < len(pred_penalties) else None
                if pred_penalty is None:
                    bad_penalty += 1
                    penalty_pred.append(-1)
                else:
                    penalty_pred.append(pred_penalty)
                penalty_weights.append(weight)

    metrics: Dict[str, Any] = {
        "article": set_metrics(
            article_gold_sets,
            article_pred_sets,
            weights=set_weights,
        ),
        "charge": set_metrics(
            charge_gold_sets,
            charge_pred_sets,
            weights=set_weights,
        ),
        "penalty": classification_metrics(
            penalty_true,
            penalty_pred,
            sample_weight=penalty_weights,
        ),
    }
    if include_summary:
        metrics["summary"] = {
            "gold_path": str(gold_path),
            "pred_path": str(pred_path),
            "pred_rows": len(pred_rows),
            "evaluated_cases": evaluated_cases,
            "evaluated_defendants": evaluated_defendants,
            "missing_gold_rows": missing_gold,
            "parse_failed_rows": parse_failed,
            "missing_defendant_predictions": missing_defendant,
            "extra_defendant_predictions": extra_defendant,
            "bad_penalty_predictions": bad_penalty,
            "alignment": "name",
        }
    return metrics


def main() -> None:
    args = parse_args()
    charge_normalizer = ChargeNormalizer(
        mapping_path=None,
        c2i_path=args.c2i_path,
        crime_encodings_path=args.crime_encodings_path,
        bge_model_path=args.bge_model_path,
        mode="multi",
    )
    metrics = evaluate(
        args.gold_path,
        args.pred_path,
        charge_normalizer=charge_normalizer,
        include_summary=args.include_summary,
    )
    print_metrics(metrics)
    maybe_write_metrics(metrics, args.output_path)


if __name__ == "__main__":
    main()
