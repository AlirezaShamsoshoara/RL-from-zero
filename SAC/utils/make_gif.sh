#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-SAC/configs/pendulum_fast.yaml}"
OUT="${OUT:-SAC/assets/sac_pendulum.gif}"
ARGS=(--config "$CONFIG" --out "$OUT")
if [[ -n "${CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
if [[ -n "${SOURCE_ROOT:-}" ]]; then ARGS+=(--source-root "$SOURCE_ROOT"); fi
if [[ -n "${SUMMARY:-}" ]]; then ARGS+=(--summary "$SUMMARY"); fi
export SDL_VIDEODRIVER=dummy
export WANDB_MODE=offline
exec "$PYTHON" -m SAC.utils.make_gif "${ARGS[@]}" "$@"
