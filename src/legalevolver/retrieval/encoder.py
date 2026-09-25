from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class SailerEncoder:
    """Shared normalized SAILER encoder used by all legal retrieval tools."""

    def __init__(self, model_path: str, device: str | None = None) -> None:
        self.model = SentenceTransformer(model_path, device=device)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.clip(norms, 1e-12, None)

