#!/usr/bin/env bash
set -euo pipefail

python -m verl.trainer.main_ppo --config-path configs --config-name grpo
