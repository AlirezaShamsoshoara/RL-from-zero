#!/usr/bin/env bash
# Regenerate the deterministic CartPole showcase from best.pt, headlessly.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-PPO/configs/cartpole_tuned.yaml}"
CHECKPOINT="${CHECKPOINT:-PPO/checkpoints_tuned/best.pt}"
OUT="${OUT:-PPO/assets/ppo_cartpole.gif}"
export SDL_VIDEODRIVER=dummy
exec "$PYTHON" -m PPO.utils.make_gif \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" --out "$OUT" "$@"
