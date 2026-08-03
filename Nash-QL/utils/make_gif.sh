#!/usr/bin/env bash
#
# Generate the Nash-QL demo GIF headlessly (the env is rendered via matplotlib,
# so no display is needed and no SDL driver is required).
#
# Runs from the repository root. The Python script puts the nash_ql package on
# sys.path itself.
#
# Usage:
#   ./Nash-QL/utils/make_gif.sh
#   PYTHON=~/.conda/envs/rlhero/bin/python ./Nash-QL/utils/make_gif.sh --episodes 4
#
# Environment overrides: PYTHON, CONFIG, CHECKPOINT, OUT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-Nash-QL/configs/grid_soccer.yaml}"
CHECKPOINT="${CHECKPOINT:-Nash-QL/checkpoints_soccer/best.pt}"
OUT="${OUT:-Nash-QL/assets/nash_ql_grid_soccer.gif}"

exec "$PYTHON" Nash-QL/utils/make_gif.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --out "$OUT" \
    "$@"
