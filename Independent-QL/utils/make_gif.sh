#!/usr/bin/env bash
#
# Generate the Independent-QL demo GIF headlessly (LineWorld is rendered via
# matplotlib, so no display is needed and no SDL driver is required).
#
# Runs from the repository root. The Python script puts the independent_ql
# package on sys.path itself.
#
# Usage:
#   ./Independent-QL/utils/make_gif.sh
#   PYTHON=~/.conda/envs/rlhero/bin/python ./Independent-QL/utils/make_gif.sh --episodes 4
#
# Environment overrides: PYTHON, CONFIG, CHECKPOINT, OUT
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-Independent-QL/configs/line_world.yaml}"
CHECKPOINT="${CHECKPOINT:-Independent-QL/checkpoints/best.pt}"
OUT="${OUT:-Independent-QL/assets/independent_ql_lineworld.gif}"

exec "$PYTHON" Independent-QL/utils/make_gif.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --out "$OUT" \
    "$@"
