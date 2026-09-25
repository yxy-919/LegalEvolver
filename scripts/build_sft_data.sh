#!/usr/bin/env bash
set -euo pipefail

python -m legalevolver.training.build_sft_data \
  --input "${TRAJECTORY_DATA_PATH:?set TRAJECTORY_DATA_PATH}" \
  --output "${SFT_DATA_PATH:?set SFT_DATA_PATH}"

