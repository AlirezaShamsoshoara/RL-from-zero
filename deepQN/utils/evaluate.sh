#!/usr/bin/env bash
# Evaluate with activated Python and the checkpoint specified by the chosen config.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-deepQN/configs/mountaincar_exploration.yaml}"
OUT="${OUT:-deepQN/logs/evaluation.json}"
EXTRA=()
[[ -n "${CHECKPOINT:-}" ]] && EXTRA+=(--checkpoint "$CHECKPOINT")
export WANDB_MODE=offline
exec "$PYTHON" -m deepQN.utils.evaluate \
    --config "$CONFIG" --out "$OUT" "${EXTRA[@]}" "$@"
