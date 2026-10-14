#!/usr/bin/env bash
# Use the caller's fresh environment. Never install, activate, log in or sync.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-TD3/configs/mountaincar_balanced.yaml}"
OUT="${OUT:-TD3/logs/evaluation.json}"
ARGS=(--config "$CONFIG" --out "$OUT")
if [[ -n "${CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "${SOURCE_ROOT:-}" ]]; then ARGS+=(--source-root "$SOURCE_ROOT"); fi
export WANDB_MODE=offline
exec "$PYTHON" -m TD3.utils.evaluate "${ARGS[@]}" "$@"
