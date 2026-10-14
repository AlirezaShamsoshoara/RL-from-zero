#!/usr/bin/env bash
# Respect the Python in the caller's fresh environment; do not install or activate.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-SAC/configs/pendulum_fast.yaml}"
OUT="${OUT:-SAC/logs/evaluation.json}"
ARGS=(--config "$CONFIG" --out "$OUT")
if [[ -n "${CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "${SOURCE_ROOT:-}" ]]; then ARGS+=(--source-root "$SOURCE_ROOT"); fi
export WANDB_MODE=offline
exec "$PYTHON" -m SAC.utils.evaluate "${ARGS[@]}" "$@"
