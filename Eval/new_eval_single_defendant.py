from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

try:
    from .common import (
        ChargeNormalizer,
        DEFAULT_BGE_MODEL_PATH,
        DEFAULT_CAIL_C2I_PATH,
        DEFAULT_CAIL_CRIME_ENCODINGS_PATH,
        DEFAULT_CHARGE_MAPPING_PATH,
        classification_metrics,
        extract_answer_payload,
        imprisonment_to_interval,
        load_jsonl,
        load_jsonl_by_case_id,
        maybe_write_metrics,
        normalize_articles,
        normalize_sentence_interval,
        print_metrics,
    )
except ImportError:
    from common import (  # type: ignore
        ChargeNormalizer,
        DEFAULT_BGE_MODEL_PATH,
        DEFAULT_CAIL_C2I_PATH,
        DEFAULT_CAIL_CRIME_ENCODINGS_PATH,
        DEFAULT_CHARGE_MAPPING_PATH,
        classification_metrics,
        extract_answer_payload,
        imprisonment_to_interval,
        load_jsonl,
        load_jsonl_by_case_id,
        maybe_write_metrics,
        normalize_articles,
        normalize_sentence_interval,
        print_metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate single-defendant SFT outputs on article, charge, and sentence interval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gold_path",
        type=Path,
        required=True,
        help="Gold CAIL-style JSONL file.",
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
        help="Include parsing counts in addition to task metrics.",
    )
    parser.add_argument(
        "--charge_mapping_path",
        type=Path,
        default=DEFAULT_CHARGE_MAPPING_PATH,
        help="Charge alias mapping JSON/JSONL path.",
    )
    parser.add_argument(
        "--c2i_path",
        type=Path,
        default=DEFAULT_CAIL_C2I_PATH,
        help="Charge vocabulary path used by BGE fallback.",
    )
    parser.add_argument(
        "--crime_encodings_path",
        type=Path,
        default=DEFAULT_CAIL_CRIME_ENCODINGS_PATH,
        help="Precomputed charge embeddings path used by BGE fallback.",
    )
    parser.add_argument(
        "--bge_model_path",
        default=DEFAULT_BGE_MODEL_PATH,
        help="BGE model path used by charge fallback matching.",
    )
    return parser.parse_args()


def gold_for_case(row: Dict[str, Any], charge_normalizer: ChargeNormalizer) -> Dict[str, Any]:
    meta = row.get("meta") or {}
    term = meta.get("term_of_imprisonment") or {}
    return {
        "article": (normalize_articles(meta.get("relevant_articles")) or ["__MISSING__"])[0],
        "charge": (charge_normalizer.normalize_pred_many(meta.get("accusation")) or ["__MISSING__"])[0],
        "penalty": imprisonment_to_interval(term),
    }


def pred_for_row(row: Dict[str, Any], charge_normalizer: ChargeNormalizer) -> Dict[str, Any]:
    payload = extract_answer_payload(str(row.get("predict") or "")) or {}
    return {
        "article": (normalize_articles(payload.get("relevant_articles")) or ["__MISSING__"])[0],
        "charge": (charge_normalizer.normalize_pred_many(payload.get("accusation")) or ["__MISSING__"])[0],
        "penalty": normalize_sentence_interval(payload.get("sentence_interval")),
        "parsed": bool(payload),
    }


def evaluate(
    gold_path: Path,
    pred_path: Path,
    *,
    charge_normalizer: ChargeNormalizer,
    include_summary: bool = False,
) -> Dict[str, Any]:
    gold_by_id = load_jsonl_by_case_id(gold_path)
    pred_rows = load_jsonl(pred_path)

    article_true = []
    article_pred = []
    charge_true = []
    charge_pred = []
    penalty_true = []
    penalty_pred = []
    missing_gold = 0
    parse_failed = 0
    bad_penalty = 0

    for pred_row in pred_rows:
        case_id = str(pred_row.get("caseID"))
        gold_row = gold_by_id.get(case_id)
        if gold_row is None:
            missing_gold += 1
            continue

        gold = gold_for_case(gold_row, charge_normalizer)
        pred = pred_for_row(pred_row, charge_normalizer)
        if not pred["parsed"]:
            parse_failed += 1

        article_true.append(gold["article"])
        article_pred.append(pred["article"])
        charge_true.append(gold["charge"])
        charge_pred.append(pred["charge"])
        penalty_true.append(gold["penalty"])
        if pred["penalty"] is None:
            bad_penalty += 1
            penalty_pred.append(-1)
        else:
            penalty_pred.append(pred["penalty"])

    metrics = {
        "article": classification_metrics(article_true, article_pred),
        "charge": classification_metrics(charge_true, charge_pred),
        "penalty": classification_metrics(penalty_true, penalty_pred),
    }
    if include_summary:
        metrics["summary"] = {
            "gold_path": str(gold_path),
            "pred_path": str(pred_path),
            "pred_rows": len(pred_rows),
            "evaluated_rows": len(article_true),
            "missing_gold_rows": missing_gold,
            "parse_failed_rows": parse_failed,
            "bad_penalty_rows": bad_penalty,
        }
    return metrics


def main() -> None:
    args = parse_args()
    charge_normalizer = ChargeNormalizer(
        mapping_path=args.charge_mapping_path,
        c2i_path=args.c2i_path,
        crime_encodings_path=args.crime_encodings_path,
        bge_model_path=args.bge_model_path,
        mode="single",
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
