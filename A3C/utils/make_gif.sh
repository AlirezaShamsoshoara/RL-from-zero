#!/usr/bin/env bash
# Complete native rollouts first; selected GIF episodes are explicitly labelled.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-A3C/configs/lunarlander_greedy.yaml}"
OUT="${OUT:-A3C/assets/a3c_lunarlander.gif}"
ARGS=(--config "$CONFIG" --out "$OUT")
if [[ -n "${CHECKPOINT:-}" ]]; then ARGS+=(--checkpoint "$CHECKPOINT"); fi
export SDL_VIDEODRIVER=dummy
export WANDB_MODE=offline
exec "$PYTHON" -m A3C.utils.make_gif "${ARGS[@]}" "$@"
