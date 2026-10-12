#!/usr/bin/env bash
# Require an explicit PPO run so another algorithm's newest run is never selected.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-PPO/assets}"
TITLE="${TITLE:-PPO CartPole-v1}"
EXTRA=()
[[ -n "${RUN:-}" ]] && EXTRA+=(--run "$RUN")
[[ -n "${SUMMARY:-}" ]] && EXTRA+=(--summary "$SUMMARY")
exec "$PYTHON" -m PPO.utils.make_charts \
    --out-dir "$OUT_DIR" --title "$TITLE" "${EXTRA[@]}" "$@"
