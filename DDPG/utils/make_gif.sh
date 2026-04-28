#!/usr/bin/env bash
#
# Generate the DDPG demo GIF headlessly (no display required).
#
# Runs from the repository root so the `DDPG.*` imports resolve, sets a dummy
# SDL video driver for off-screen Box2D/pygame rendering, and calls make_gif.py
# with sensible defaults (the tuned LunarLanderContinuous policy).
#
# Usage:
#   ./DDPG/utils/make_gif.sh                       # regenerate the default demo
#   ./DDPG/utils/make_gif.sh --episodes 12 --keep-top 2   # forward extra flags
#   CHECKPOINT=DDPG/checkpoints/best_01.pt ./DDPG/utils/make_gif.sh
#   PYTHON=~/.conda/envs/rlhero/bin/python ./DDPG/utils/make_gif.sh
#
# Environment overrides: PYTHON, CONFIG, CHECKPOINT, OUT, SDL_VIDEODRIVER
set -euo pipefail

# Resolve repo root (this script lives in DDPG/utils/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Headless rendering for Box2D/pygame — renders to an off-screen surface.
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-DDPG/configs/lunarlander_continuous_tuned.yaml}"
CHECKPOINT="${CHECKPOINT:-DDPG/checkpoints_tuned/best.pt}"
OUT="${OUT:-DDPG/assets/ddpg_lunarlander.gif}"

exec "$PYTHON" -m DDPG.utils.make_gif \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --out "$OUT" \
    "$@"
