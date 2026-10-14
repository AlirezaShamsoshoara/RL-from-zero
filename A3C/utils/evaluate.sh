#!/usr/bin/env bash
# Use the Python selected by the caller's fresh activated environment.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-A3C/configs/lunarlander_greedy.yaml}"
OUT="${OUT:-A3C/logs/evaluation.json}"
ARGS=(--config "$CONFIG" --out "$OUT")
if [[ -n "${CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
export WANDB_MODE=offline
exec "$PYTHON" -m A3C.utils.evaluate "${ARGS[@]}" "$@"
