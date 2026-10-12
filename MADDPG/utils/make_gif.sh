#!/usr/bin/env bash
#
# Generate the MADDPG demo GIF headlessly (no display required).
#
# Runs from the repository root so the `MADDPG.*` imports resolve, sets a dummy
# SDL video driver for off-screen pygame rendering, and calls make_gif.py with
# sensible defaults (the trained simple_spread policy).
#
# Usage:
#   ./MADDPG/utils/make_gif.sh                                # regenerate the default demo
#   ./MADDPG/utils/make_gif.sh --episodes 20 --keep-top 3    # forward extra flags
#   CHECKPOINT=MADDPG/checkpoints/checkpoint_50000.pt ./MADDPG/utils/make_gif.sh
#   PYTHON=~/.conda/envs/rlhero/bin/python ./MADDPG/utils/make_gif.sh
#
# Environment overrides: PYTHON, CONFIG, CHECKPOINT, OUT, SDL_VIDEODRIVER
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Headless rendering for pygame - renders to an off-screen surface.
export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-MADDPG/configs/simple_spread.yaml}"
CHECKPOINT="${CHECKPOINT:-MADDPG/checkpoints/best.pt}"
OUT="${OUT:-MADDPG/assets/maddpg_simple_spread.gif}"

exec "$PYTHON" -m MADDPG.utils.make_gif \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --out "$OUT" \
    "$@"
