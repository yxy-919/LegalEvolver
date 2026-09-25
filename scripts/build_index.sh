#!/usr/bin/env bash
set -euo pipefail

python -m legalevolver.retrieval.build_index \
  --documents "${RETRIEVAL_DOCUMENTS_PATH:?set RETRIEVAL_DOCUMENTS_PATH}" \
  --text-field "${RETRIEVAL_TEXT_FIELD:-text}" \
  --encoder "${SAILER_MODEL_PATH:?set SAILER_MODEL_PATH}" \
  --index "${RETRIEVAL_INDEX_PATH:?set RETRIEVAL_INDEX_PATH}" \
  --metadata "${RETRIEVAL_METADATA_PATH:?set RETRIEVAL_METADATA_PATH}"
