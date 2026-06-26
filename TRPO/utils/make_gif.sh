#!/usr/bin/env bash
#
# Generate the TRPO demo GIF headlessly (no display required).
#
# Runs from the repository root so the TRPO.* imports resolve, sets a dummy SDL
# video driver for off-screen rendering, and calls make_gif.py with defaults.
# Works for both continuous (BipedalWalker) and discrete (Acrobot) checkpoints.
#
# Usage:
#   ./TRPO/utils/make_gif.sh                                   # default (BipedalWalker)
#   CONFIG=TRPO/configs/acrobot.yaml \
#     CHECKPOINT=TRPO/checkpoints_acrobot/best.pt \
#     OUT=TRPO/assets/trpo_acrobot.gif ./TRPO/utils/make_gif.sh
#   PYTHON=~/.conda/envs/rlhero/bin/python ./TRPO/utils/make_gif.sh
#
# Environment overrides: PYTHON, CONFIG, CHECKPOINT, OUT, SDL_VIDEODRIVER
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

export SDL_VIDEODRIVER="${SDL_VIDEODRIVER:-dummy}"

PYTHON="${PYTHON:-python}"
# Defaults render the Acrobot showcase (the better-performing run). Acrobot is a
# discrete/categorical policy, so STOCHASTIC defaults on (greedy argmax can stall).
CONFIG="${CONFIG:-TRPO/configs/acrobot.yaml}"
CHECKPOINT="${CHECKPOINT:-TRPO/checkpoints_acrobot/best.pt}"
OUT="${OUT:-TRPO/assets/trpo_acrobot.gif}"
STOCHASTIC="${STOCHASTIC:-1}"

EXTRA=()
[[ "$STOCHASTIC" == "1" ]] && EXTRA+=(--stochastic)

exec "$PYTHON" -m TRPO.utils.make_gif \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --out "$OUT" \
    "${EXTRA[@]}" \
    "$@"
