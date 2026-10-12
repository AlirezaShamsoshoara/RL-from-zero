#!/usr/bin/env bash
# Evaluate a checkpoint using the Python from the caller's activated environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-PPO/configs/cartpole_tuned.yaml}"
CHECKPOINT="${CHECKPOINT:-PPO/checkpoints_tuned/best.pt}"
OUT="${OUT:-PPO/logs/evaluation.json}"
export WANDB_MODE=offline
exec "$PYTHON" -m PPO.utils.evaluate \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --out "$OUT" "$@"
