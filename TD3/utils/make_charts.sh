#!/usr/bin/env bash
# RUN must name the exact closed producer datastore, not a newest-run glob.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-TD3/configs/mountaincar_balanced.yaml}"
OUT_DIR="${OUT_DIR:-TD3/assets}"
TITLE="${TITLE:-TD3 MountainCarContinuous-v0}"
ARGS=(--config "$CONFIG" --out-dir "$OUT_DIR" --title "$TITLE")
if [[ -n "${RUN:-}" ]]; then ARGS+=(--run "$RUN"); fi
if [[ -n "${SUMMARY:-}" ]]; then ARGS+=(--summary "$SUMMARY"); fi
if [[ -n "${EVALUATION:-}" ]]; then ARGS+=(--evaluation "$EVALUATION"); fi
export WANDB_MODE=offline
exec "$PYTHON" -m TD3.utils.make_charts "${ARGS[@]}" "$@"
