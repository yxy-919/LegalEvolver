#!/usr/bin/env bash
set -euo pipefail

: "${BASE_MODEL:?set BASE_MODEL}"
: "${SFT_DATA:?set SFT_DATA}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"

llamafactory-cli train \
  --stage sft \
  --do_train true \
  --model_name_or_path "$BASE_MODEL" \
  --dataset "$SFT_DATA" \
  --output_dir "$OUTPUT_DIR" \
  --finetuning_type lora \
  --cutoff_len 12288 \
  --learning_rate 8e-5 \
  --num_train_epochs 2 \
  --bf16 true \
  --flash_attn sdpa \
  "$@"
