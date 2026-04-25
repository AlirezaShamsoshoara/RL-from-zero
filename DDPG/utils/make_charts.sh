#!/usr/bin/env bash
#
# Generate the DDPG training charts (chart_01/02/03.png) from a local wandb run.
#
# Runs from the repository root so the wandb datastore reader and output paths
# resolve. By default it uses the newest run under wandb/ - pass RUN to pick one.
#
# Usage:
#   ./DDPG/utils/make_charts.sh                                  # newest wandb run
#   RUN=wandb/offline-run-XXXX/run-XXXX.wandb ./DDPG/utils/make_charts.sh
#   TITLE="pure DDPG (collapsed)" ./DDPG/utils/make_charts.sh
#   PYTHON=~/.conda/envs/rlhero/bin/python ./DDPG/utils/make_charts.sh
#
# Environment overrides: PYTHON, RUN, OUT_DIR, TITLE
set -euo pipefail

# Resolve repo root (this script lives in DDPG/utils/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-DDPG/assets}"
TITLE="${TITLE:-DDPG + target smoothing}"

# Default RUN = the most recently modified .wandb file under wandb/.
if [[ -z "${RUN:-}" ]]; then
    RUN="$(ls -t wandb/*/run-*.wandb 2>/dev/null | head -1 || true)"
    if [[ -z "$RUN" ]]; then
        echo "No .wandb run found under wandb/. Set RUN=<path to run-*.wandb>." >&2
        exit 1
    fi
    echo "Using newest run: $RUN"
fi

exec "$PYTHON" -m DDPG.utils.make_charts \
    --run "$RUN" \
    --out-dir "$OUT_DIR" \
    --title "$TITLE" \
    "$@"
