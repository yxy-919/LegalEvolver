from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


class DenseIndex:
    def __init__(self, index_path: Path, metadata_path: Path) -> None:
        self.index = faiss.read_index(str(index_path))
        with metadata_path.open(encoding="utf-8") as handle:
            self.metadata = [json.loads(line) for line in handle if line.strip()]

    def search(self, query_vector: np.ndarray, candidate_count: int) -> list[tuple[float, dict[str, Any]]]:
        scores, indices = self.index.search(query_vector.astype("float32"), candidate_count)
        return [
            (float(score), self.metadata[int(index)])
            for score, index in zip(scores[0], indices[0])
            if 0 <= int(index) < len(self.metadata)
        ]

