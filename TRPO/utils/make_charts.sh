#!/usr/bin/env bash
#
# Generate the TRPO training charts (chart_01/02/03.png) from a local wandb run.
#
# Runs from the repository root. By default it uses the newest run under wandb/;
# pass RUN to pick a specific one.
#
# Usage:
#   ./TRPO/utils/make_charts.sh                                  # newest wandb run
#   RUN=wandb/offline-run-XXXX/run-XXXX.wandb TITLE="TRPO BipedalWalker-v3" \
#     SOLVED=300 ./TRPO/utils/make_charts.sh
#   PYTHON=~/.conda/envs/rlhero/bin/python ./TRPO/utils/make_charts.sh
#
# Environment overrides: PYTHON, RUN, OUT_DIR, TITLE, SOLVED, MAX_KL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-TRPO/assets}"
TITLE="${TITLE:-TRPO}"
MAX_KL="${MAX_KL:-0.01}"

if [[ -z "${RUN:-}" ]]; then
    RUN="$(ls -t wandb/*/run-*.wandb 2>/dev/null | head -1 || true)"
    if [[ -z "$RUN" ]]; then
        echo "No .wandb run found under wandb/. Set RUN=<path to run-*.wandb>." >&2
        exit 1
    fi
    echo "Using newest run: $RUN"
fi

EXTRA=()
[[ -n "${SOLVED:-}" ]] && EXTRA+=(--solved "$SOLVED")
[[ -n "${MAX_KL:-}" ]] && EXTRA+=(--max-kl "$MAX_KL")

exec "$PYTHON" -m TRPO.utils.make_charts \
    --run "$RUN" \
    --out-dir "$OUT_DIR" \
    --title "$TITLE" \
    "${EXTRA[@]}" \
    "$@"
