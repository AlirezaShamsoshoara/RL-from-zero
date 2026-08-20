#!/usr/bin/env bash
#
# Generate the Nash-QL training charts (chart_01/02/03.png) from a wandb run.
#
# Runs from the repository root. By default it uses the newest run under wandb/;
# pass RUN to pick a specific one.
#
# Usage:
#   ./Nash-QL/utils/make_charts.sh                          # newest wandb run
#   RUN=wandb/offline-run-XXXX/run-XXXX.wandb TITLE="Nash-QL Grid Soccer" \
#     ./Nash-QL/utils/make_charts.sh
#   PYTHON=~/.conda/envs/rlhero/bin/python ./Nash-QL/utils/make_charts.sh
#
# Environment overrides: PYTHON, RUN, OUT_DIR, TITLE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-Nash-QL/assets}"
TITLE="${TITLE:-Nash-QL Grid Soccer}"

if [[ -z "${RUN:-}" ]]; then
    RUN="$(ls -t wandb/*/run-*.wandb 2>/dev/null | head -1 || true)"
    if [[ -z "$RUN" ]]; then
        echo "No .wandb run found under wandb/. Set RUN=<path to run-*.wandb>." >&2
        exit 1
    fi
    echo "Using newest run: $RUN"
fi

exec "$PYTHON" Nash-QL/utils/make_charts.py \
    --run "$RUN" \
    --out-dir "$OUT_DIR" \
    --title "$TITLE" \
    "$@"
