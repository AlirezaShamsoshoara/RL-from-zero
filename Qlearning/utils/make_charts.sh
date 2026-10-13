#!/usr/bin/env bash
# Select an explicit Qlearning training run, never another algorithm's newest run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-Qlearning/assets}"
TITLE="${TITLE:-Q-learning FrozenLake}"
EXTRA=()
[[ -n "${RUN:-}" ]] && EXTRA+=(--run "$RUN")
[[ -n "${SUMMARY:-}" ]] && EXTRA+=(--summary "$SUMMARY")
exec "$PYTHON" -m Qlearning.utils.make_charts \
    --out-dir "$OUT_DIR" --title "$TITLE" "${EXTRA[@]}" "$@"
