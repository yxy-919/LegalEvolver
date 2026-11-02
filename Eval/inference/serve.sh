#!/usr/bin/env bash
set -euo pipefail
MODEL_PATH="${1:?Usage: bash Eval/inference/serve.sh /path/to/LegalEvolver-Qwen3-8B}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name LegalEvolver \
  --host 127.0.0.1 --port "${PORT:-8000}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
  --max-model-len "${MAX_MODEL_LEN:-40960}"
