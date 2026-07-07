#!/usr/bin/env bash
#
# Generate the IQL demo GIF headlessly (no display required).
#
# Runs from the repository root so the IQL.* imports resolve, sets a dummy SDL
# video driver for off-screen rendering, and calls make_gif.py with defaults
# (the mixed-dataset Pendulum policy, which is the showcase).
#
# Usage:
#   ./IQL/utils/make_gif.sh                                    # default (mixed)
#   CONFIG=IQL/configs/pendulum_random.yaml \
#     CHECKPOINT=IQL/checkpoints_random/best.pt \
#     OUT=IQL/assets/iql_pendulum_random.gif ./IQL/utils/make_gif.sh
#   PYTHON=~/.conda/envs/rlhero/bin/python ./IQL/utils/make_gif.sh
#
# Environment overrides: PYTHON, CONFIG, CHECKPOINT, OUT, SDL_VIDEODRIVER
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-IQL/configs/pendulum_mixed.yaml}"
CHECKPOINT="${CHECKPOINT:-IQL/checkpoints_mixed/best.pt}"
OUT="${OUT:-IQL/assets/iql_pendulum.gif}"

exec "$PYTHON" -m IQL.utils.make_gif \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --out "$OUT" \
    "$@"
