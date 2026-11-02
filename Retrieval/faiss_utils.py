from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import faiss
import numpy as np

from Retrieval.common.io import ensure_parent_dir, read_jsonl, write_jsonl


def build_ip_index(embeddings: np.ndarray) -> faiss.Index:
    matrix = np.ascontiguousarray(embeddings.astype("float32"))
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def save_index(index: faiss.Index, path: str | Path) -> None:
    ensure_parent_dir(path)
    faiss.write_index(index, str(path))


def load_index(path: str | Path) -> faiss.Index:
    return faiss.read_index(str(path))


def search_index(index: faiss.Index, query_vector: np.ndarray, topk: int) -> Tuple[np.ndarray, np.ndarray]:
    vector = np.ascontiguousarray(query_vector.astype("float32"))
    scores, indices = index.search(vector, topk)
    return scores, indices


def save_metadata(rows: Iterable[Dict], path: str | Path) -> None:
    write_jsonl(path, rows)


def load_metadata(path: str | Path) -> List[Dict]:
    return read_jsonl(path)
