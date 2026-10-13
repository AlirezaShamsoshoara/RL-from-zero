#!/usr/bin/env bash
# Render a slow FrozenLake showcase using the checkpoint named in the selected config.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-Qlearning/configs/frozenlake.yaml}"
OUT="${OUT:-Qlearning/assets/qlearning_frozenlake_8x8.gif}"
EXTRA=()
[[ -n "${CHECKPOINT:-}" ]] && EXTRA+=(--checkpoint "$CHECKPOINT")
export SDL_VIDEODRIVER=dummy
exec "$PYTHON" -m Qlearning.utils.make_gif \
    --config "$CONFIG" --out "$OUT" "${EXTRA[@]}" "$@"
