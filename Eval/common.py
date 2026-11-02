from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Sequence

EVAL_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CHARGE_MAPPING_PATH = EVAL_DATA_DIR / "single_expanded_charges_mapping.json"
DEFAULT_CAIL_C2I_PATH = EVAL_DATA_DIR / "single_c2i.json"
DEFAULT_CAIL_CRIME_ENCODINGS_PATH = EVAL_DATA_DIR / "single_crime_encodings.npy"
DEFAULT_CMDL_C2I_PATH = EVAL_DATA_DIR / "multi_c2i.json"
DEFAULT_CMDL_CRIME_ENCODINGS_PATH = EVAL_DATA_DIR / "multi_crime_encodings.npy"
DEFAULT_C2I_PATH = DEFAULT_CAIL_C2I_PATH
DEFAULT_CRIME_ENCODINGS_PATH = DEFAULT_CAIL_CRIME_ENCODINGS_PATH
DEFAULT_BGE_MODEL_PATH = os.environ.get("BGE_MODEL_PATH", "")

CHINESE_TO_NUM = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
    "百": 100,
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_jsonl_by_case_id(path: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for row in load_jsonl(path):
        case_id = row.get("caseID")
        if case_id is not None:
            rows[str(case_id)] = row
    return rows


def chinese_to_arabic(text: str) -> int:
    result = 0
    unit = 1
    for char in reversed(text):
        if char not in CHINESE_TO_NUM:
            continue
        num = CHINESE_TO_NUM[char]
        if num >= 10:
            unit = num
        else:
            result += num * unit
    return result


def normalize_article(article: Any, *, keep_clause: bool = False) -> str | None:
    if article is None:
        return None
    text = str(article).strip()
    if not text:
        return None
    text = text.replace("條", "条")
    match = re.search(r"第\s*(\d+(?:-\d+)?|[一二三四五六七八九十百零]+)\s*条", text)
    if match:
        value = match.group(1)
        if value.isdigit():
            return str(int(value))
        if "-" in value:
            return value if keep_clause else value.split("-", 1)[0]
        return str(chinese_to_arabic(value))
    match = re.search(r"(\d+(?:-\d+)?)", text)
    if match:
        value = match.group(1)
        if "-" in value and not keep_clause:
            value = value.split("-", 1)[0]
        return str(int(value)) if value.isdigit() else value
    return text


def normalize_articles(articles: Any, *, keep_clause: bool = False) -> List[str]:
    if articles is None:
        return []
    if isinstance(articles, (str, int)):
        candidates: Iterable[Any] = [articles]
    elif isinstance(articles, list):
        candidates = articles
    else:
        candidates = [articles]

    normalized: List[str] = []
    for item in candidates:
        if isinstance(item, str):
            parts = re.split(r"[、,，;；\s]+", item)
        else:
            parts = [item]
        for part in parts:
            article = normalize_article(part, keep_clause=keep_clause)
            if article is not None:
                normalized.append(article)
    return sorted(set(normalized), key=lambda value: (int(value) if value.isdigit() else 10**9, value))


def normalize_charge(charge: Any, *, keep_suffix: bool = True) -> str | None:
    if charge is None:
        return None
    text = str(charge).strip()
    if not text:
        return None
    text = re.sub(r"\s+", "", text)
    if keep_suffix:
        return text if text.endswith("罪") else f"{text}罪"
    return text[:-1] if text.endswith("罪") else text


def normalize_charges(charges: Any) -> List[str]:
    if charges is None:
        return []
    if isinstance(charges, str):
        candidates = re.split(
            r"(?<=罪)[、,，;；和及与]+(?=.+?罪)",
            charges,
        )
    elif isinstance(charges, list):
        candidates = charges
    else:
        candidates = [charges]

    normalized = [normalize_charge(item) for item in candidates]
    return sorted({item for item in normalized if item})


def normalize_gold_charge(charge: Any) -> str | None:
    if charge is None:
        return None
    text = str(charge).strip()
    if not text:
        return None
    return re.sub(r"\s+", "", text)


def normalize_gold_charges(charges: Any) -> List[str]:
    if charges is None:
        return []
    if isinstance(charges, list):
        candidates = charges
    else:
        candidates = [charges]
    normalized = [normalize_gold_charge(item) for item in candidates]
    return sorted({item for item in normalized if item})


def load_mapping_dict(path: Path | None) -> Dict[str, str]:
    if path is None or not path.exists():
        return {}
    mapping: Dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {str(key): str(value) for key, value in data.items()}
    except json.JSONDecodeError:
        pass

    for line in text.splitlines():
        if not line.strip():
            continue
        data = json.loads(line)
        if isinstance(data, dict):
            mapping.update({str(key): str(value) for key, value in data.items()})
    return mapping


class ChargeNormalizer:
    def __init__(
        self,
        *,
        mapping_path: Path | None = DEFAULT_CHARGE_MAPPING_PATH,
        c2i_path: Path | None = DEFAULT_C2I_PATH,
        crime_encodings_path: Path | None = DEFAULT_CRIME_ENCODINGS_PATH,
        bge_model_path: str | None = DEFAULT_BGE_MODEL_PATH,
        mode: Literal["single", "multi"] = "single",
    ) -> None:
        self.mode = mode
        self.mapping_dict = load_mapping_dict(mapping_path)
        self._crime_names: List[str] = []
        if c2i_path is not None and c2i_path.exists():
            data = json.loads(c2i_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._crime_names = [normalize_gold_charge(key) or str(key) for key in data]

        self._bge_model_path = bge_model_path
        self._crime_encodings_path = crime_encodings_path
        self._bge_model = None
        self._crime_embeddings = None

    def normalize_gold_one(self, charge: Any) -> str | None:
        return normalize_gold_charge(charge)

    def normalize_pred_one(self, charge: Any) -> str | None:
        if self.mode == "multi":
            normalized = normalize_charge(charge, keep_suffix=True)
            return self._bge_match(normalized) or normalized

        normalized = normalize_charge(charge, keep_suffix=False)
        if normalized is None:
            return None
        mapped = self.mapping_dict.get(normalized, normalized)
        mapped = normalize_charge(mapped, keep_suffix=False)
        if mapped in self._crime_names:
            return mapped
        return self._bge_match(mapped) or mapped

    def normalize_gold_many(self, charges: Any) -> List[str]:
        return normalize_gold_charges(charges)

    def normalize_pred_many(self, charges: Any) -> List[str]:
        return sorted(
            {
                normalized
                for charge in normalize_charges(charges)
                for normalized in [self.normalize_pred_one(charge)]
                if normalized
            }
        )

    def normalize_one(self, charge: Any) -> str | None:
        return self.normalize_pred_one(charge)

    def normalize_many(self, charges: Any) -> List[str]:
        return self.normalize_pred_many(charges)

    def _bge_match(self, charge: str) -> str | None:
        if (
            not self._bge_model_path
            or self._crime_encodings_path is None
            or not self._crime_encodings_path.exists()
            or not self._crime_names
        ):
            return None
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            return None

        if self._bge_model is None:
            self._bge_model = SentenceTransformer(self._bge_model_path)
        if self._crime_embeddings is None:
            self._crime_embeddings = np.load(self._crime_encodings_path)

        embedding = self._bge_model.encode([charge])
        similarities = cosine_similarity(embedding, self._crime_embeddings)
        best_idx = int(np.argmax(similarities[0]))
        if best_idx < len(self._crime_names):
            return self._crime_names[best_idx]
        return None


def imprisonment_to_interval(penalty: Dict[str, Any] | int | float | None) -> int:
    if isinstance(penalty, dict):
        if penalty.get("death_penalty") or penalty.get("life_imprisonment"):
            return 0
        months = int(penalty.get("imprisonment") or 0)
    else:
        months = int(penalty or 0)

    if months > 20 * 12:
        return 0
    if months > 10 * 12:
        return 1
    if months > 7 * 12:
        return 2
    if months > 5 * 12:
        return 3
    if months > 3 * 12:
        return 4
    if months > 2 * 12:
        return 5
    if months > 1 * 12:
        return 6
    if months > 9:
        return 7
    if months > 6:
        return 8
    if months > 0:
        return 9
    return 10


def normalize_sentence_interval(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if 0 <= value <= 10 else None
    text = str(value).strip()
    if not text:
        return None
    match = re.fullmatch(r"\d+", text)
    if match:
        label = int(text)
        return label if 0 <= label <= 10 else None
    match = re.search(r"(\d+)", text)
    if match:
        label = int(match.group(1))
        return label if 0 <= label <= 10 else None
    return None


def extract_answer_payload(text: str) -> Dict[str, Any] | None:
    text = (text or "").split("</think>")[-1]
    matches = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S)
    candidate = matches[-1] if matches else text
    candidate = candidate.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.S)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else None
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        try:
            payload = json.loads(candidate[start : end + 1])
            return payload if isinstance(payload, dict) else None
        except (json.JSONDecodeError, ValueError, TypeError):
            return None
    return None


def classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    sample_weight: Sequence[float] | None = None,
) -> Dict[str, float]:
    if not y_true:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    if sample_weight is None:
        sample_weight = [1.0] * len(y_true)

    total_weight = float(sum(sample_weight))
    correct_weight = sum(
        weight
        for gold, pred, weight in zip(y_true, y_pred, sample_weight)
        if gold == pred
    )
    labels = sorted(set(y_true) | set(y_pred), key=str)
    precisions = []
    recalls = []
    f1s = []
    for label in labels:
        tp = fp = fn = 0.0
        for gold, pred, weight in zip(y_true, y_pred, sample_weight):
            if pred == label and gold == label:
                tp += weight
            elif pred == label and gold != label:
                fp += weight
            elif pred != label and gold == label:
                fn += weight
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return {
        "accuracy": correct_weight / total_weight if total_weight else 0.0,
        "precision": sum(precisions) / len(precisions) if precisions else 0.0,
        "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }


def set_metrics(
    gold_sets: Sequence[set[str]],
    pred_sets: Sequence[set[str]],
    *,
    weights: Sequence[float] | None = None,
) -> Dict[str, float]:
    if not gold_sets:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    if weights is None:
        weights = [1.0] * len(gold_sets)

    total_weight = float(sum(weights))
    exact = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    for gold, pred, weight in zip(gold_sets, pred_sets, weights):
        true_positive = len(gold & pred)
        precision = true_positive / len(pred) if pred else 0.0
        recall = true_positive / len(gold) if gold else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        exact += weight * float(gold == pred)
        precision_sum += weight * precision
        recall_sum += weight * recall
        f1_sum += weight * f1

    return {
        "accuracy": exact / total_weight if total_weight else 0.0,
        "precision": precision_sum / total_weight if total_weight else 0.0,
        "recall": recall_sum / total_weight if total_weight else 0.0,
        "f1": f1_sum / total_weight if total_weight else 0.0,
    }


def per_defendant_weight(defendant_count: int) -> float:
    if defendant_count <= 1:
        return 1.0
    import math

    return math.log(defendant_count, 2) / defendant_count


def print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def maybe_write_metrics(metrics: Dict[str, Any], output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
