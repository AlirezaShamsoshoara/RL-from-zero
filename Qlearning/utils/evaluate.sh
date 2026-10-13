#!/usr/bin/env bash
# Evaluate with the caller's active Python; checkpoint defaults come from the config.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-Qlearning/configs/frozenlake.yaml}"
OUT="${OUT:-Qlearning/logs/evaluation.json}"
EXTRA=()
[[ -n "${CHECKPOINT:-}" ]] && EXTRA+=(--checkpoint "$CHECKPOINT")
export WANDB_MODE=offline
exec "$PYTHON" -m Qlearning.utils.evaluate \
    --config "$CONFIG" --out "$OUT" "${EXTRA[@]}" "$@"
