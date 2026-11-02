from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class EncoderConfig:
    model_path: str
    batch_size: int = 64
    max_length: int = 512
    device: Optional[str] = None


class SailerEncoder:
    def __init__(self, model_path: str, *, batch_size: int = 64, max_length: int = 512, device: str | None = None):
        self.config = EncoderConfig(model_path=model_path, batch_size=batch_size, max_length=max_length, device=device)
        self.model_path = model_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or self._default_device()
        self.backend = ""
        self._st_model = None
        self._tokenizer = None
        self._hf_model = None
        self._load_model()

    @staticmethod
    def _default_device() -> str:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def normalize(embeddings: np.ndarray) -> np.ndarray:
        matrix = np.asarray(embeddings, dtype=np.float32)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return matrix / norms

    def _load_model(self) -> None:
        model_dir = Path(self.model_path)
        if (model_dir / "modules.json").exists():
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(self.model_path, device=self.device)
                self.backend = "sentence_transformers"
                return
            except Exception:
                self._st_model = None

        from transformers import AutoModel, AutoTokenizer

        import torch

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self._hf_model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        self._hf_model.to(self.device)
        self._hf_model.eval()
        if self.device.startswith("cuda"):
            self._hf_model = self._hf_model.to(dtype=torch.bfloat16)
        self.backend = "transformers"

    def _encode_with_sentence_transformers(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self._st_model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return self.normalize(embeddings)

    def _encode_with_transformers(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        outputs: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                model_output = self._hf_model(**encoded)
                if hasattr(model_output, "last_hidden_state"):
                    token_embeddings = model_output.last_hidden_state
                else:
                    token_embeddings = model_output[0]
                attention_mask = encoded["attention_mask"].unsqueeze(-1)
                masked = token_embeddings * attention_mask
                summed = masked.sum(dim=1)
                counts = attention_mask.sum(dim=1).clamp(min=1)
                pooled = summed / counts
                pooled = F.normalize(pooled.float(), p=2, dim=1)
            outputs.append(pooled.detach().cpu().numpy().astype(np.float32))
        return self.normalize(np.vstack(outputs) if outputs else np.zeros((0, 1), dtype=np.float32))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        normalized_texts = [str(text or "").strip() for text in texts]
        if not normalized_texts:
            return np.zeros((0, 0), dtype=np.float32)
        if self.backend == "sentence_transformers":
            return self._encode_with_sentence_transformers(normalized_texts)
        return self._encode_with_transformers(normalized_texts)
