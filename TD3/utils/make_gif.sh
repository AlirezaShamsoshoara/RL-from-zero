#!/usr/bin/env bash
# Headless rendering only when the caller deliberately executes this wrapper.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-TD3/configs/mountaincar_balanced.yaml}"
OUT="${OUT:-TD3/assets/td3_mountaincar_continuous.gif}"
ARGS=(--config "$CONFIG" --out "$OUT")
if [[ -n "${CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "${SOURCE_ROOT:-}" ]]; then ARGS+=(--source-root "$SOURCE_ROOT"); fi
if [[ -n "${SUMMARY:-}" ]]; then ARGS+=(--summary "$SUMMARY"); fi
export SDL_VIDEODRIVER=dummy
export WANDB_MODE=offline
exec "$PYTHON" -m TD3.utils.make_gif "${ARGS[@]}" "$@"
