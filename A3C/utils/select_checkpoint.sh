#!/usr/bin/env bash
# Select one profile using development seeds, never implicit held-out test data.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
ARGS=()
if [[ -n "${CONFIG:-}" ]]; then ARGS+=(--config "$CONFIG"); fi
if [[ -n "${CHECKPOINT_DIR:-}" ]]; then ARGS+=(--checkpoint-dir "$CHECKPOINT_DIR"); fi
if [[ -n "${OUT:-}" ]]; then ARGS+=(--out "$OUT"); fi
if [[ -n "${SUMMARY:-}" ]]; then ARGS+=(--summary "$SUMMARY"); fi
export WANDB_MODE=offline
exec "$PYTHON" -m A3C.utils.select_checkpoint "${ARGS[@]}" "$@"
