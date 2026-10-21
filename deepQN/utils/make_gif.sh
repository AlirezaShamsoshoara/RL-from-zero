#!/usr/bin/env bash
# Render a deterministic showcase, inferring the checkpoint from the chosen config.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-deepQN/configs/mountaincar_exploration.yaml}"
OUT="${OUT:-deepQN/assets/dqn_mountaincar.gif}"
EXTRA=()
[[ -n "${CHECKPOINT:-}" ]] && EXTRA+=(--checkpoint "$CHECKPOINT")
export SDL_VIDEODRIVER=dummy
exec "$PYTHON" -m deepQN.utils.make_gif \
    --config "$CONFIG" --out "$OUT" "${EXTRA[@]}" "$@"
