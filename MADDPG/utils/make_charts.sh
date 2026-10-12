#!/usr/bin/env bash
#
# Generate the MADDPG training charts (chart_01/02/03.png) from a local wandb run.
#
# Runs from the repository root so the wandb datastore reader and output paths
# resolve. By default it uses the newest run under wandb/ - pass RUN to pick one.
#
# Usage:
#   ./MADDPG/utils/make_charts.sh                              # newest wandb run
#   RUN=wandb/offline-run-XXXX/run-XXXX.wandb ./MADDPG/utils/make_charts.sh
#   TITLE="MADDPG (simple_spread)" ./MADDPG/utils/make_charts.sh
#   PYTHON=~/.conda/envs/rlhero/bin/python ./MADDPG/utils/make_charts.sh
#
# Environment overrides: PYTHON, RUN, OUT_DIR, TITLE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-MADDPG/assets}"
TITLE="${TITLE:-MADDPG on simple_spread_v3}"

if [[ -z "${RUN:-}" ]]; then
    RUN="$(ls -t wandb/*/run-*.wandb 2>/dev/null | head -1 || true)"
    if [[ -z "$RUN" ]]; then
        echo "No .wandb run found under wandb/. Set RUN=<path to run-*.wandb>." >&2
        exit 1
    fi
    echo "Using newest run: $RUN"
fi

exec "$PYTHON" -m MADDPG.utils.make_charts \
    --run "$RUN" \
    --out-dir "$OUT_DIR" \
    --title "$TITLE" \
    "$@"
