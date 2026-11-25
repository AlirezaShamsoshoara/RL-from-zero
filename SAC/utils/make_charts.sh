#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-SAC/configs/pendulum_fast.yaml}"
OUT_DIR="${OUT_DIR:-SAC/assets}"
TITLE="${TITLE:-SAC Pendulum-v1}"
ARGS=(--config "$CONFIG" --out-dir "$OUT_DIR" --title "$TITLE")
# --run remains required. No newest-run discovery or baseline substitution.
if [[ -n "${RUN:-}" ]]; then ARGS+=(--run "$RUN"); fi
if [[ -n "${SUMMARY:-}" ]]; then ARGS+=(--summary "$SUMMARY"); fi
if [[ -n "${EVALUATION:-}" ]]; then ARGS+=(--evaluation "$EVALUATION"); fi
export WANDB_MODE=offline
exec "$PYTHON" -m SAC.utils.make_charts "${ARGS[@]}" "$@"
