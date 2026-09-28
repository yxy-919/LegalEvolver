#!/usr/bin/env bash
set -euo pipefail

: "${SFT_CHECKPOINT:?set SFT_CHECKPOINT}"
: "${GRPO_DATA:?set GRPO_DATA}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${JUDGE_BASE_URL:?set JUDGE_BASE_URL}"
: "${RUBRIC_BANK_PATH:=rubric_generation/final_rubric_bank.jsonl}"

export JUDGE_MODEL="${JUDGE_MODEL:-judge-model}"
export JUDGE_API_KEY="${JUDGE_API_KEY:-EMPTY}"
export RUBRIC_BANK_PATH
export OUTCOME_REWARD_WEIGHT="${OUTCOME_REWARD_WEIGHT:-0.891}"
export FORMAT_REWARD_WEIGHT="${FORMAT_REWARD_WEIGHT:-0.099}"
export RUBRIC_REWARD_WEIGHT="${RUBRIC_REWARD_WEIGHT:-0.01}"

python -m verl.trainer.main_ppo \
  data.train_files="$GRPO_DATA" \
  actor_rollout_ref.model.path="$SFT_CHECKPOINT" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.max_tool_calls=10 \
  custom_reward_function.path=grpo/reward.py \
  "$@"
