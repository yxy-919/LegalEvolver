from __future__ import annotations

import argparse
import json
from pathlib import Path

import faiss

from .encoder import SailerEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()

    with args.documents.open(encoding="utf-8") as handle:
        documents = [json.loads(line) for line in handle if line.strip()]
    vectors = SailerEncoder(args.encoder).encode([str(item[args.text_field]) for item in documents])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype("float32"))

    args.index.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(args.index))
    with args.metadata.open("w", encoding="utf-8") as handle:
        for item in documents:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

