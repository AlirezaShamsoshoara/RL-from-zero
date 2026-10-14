#!/usr/bin/env bash
# Require an explicit A3C training datastore; do not select a latest run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-A3C/configs/lunarlander.yaml}"
OUT_DIR="${OUT_DIR:-A3C/assets}"
TITLE="${TITLE:-A3C LunarLander-v3}"
ARGS=(--config "$CONFIG" --out-dir "$OUT_DIR" --title "$TITLE")
if [[ -n "${RUN:-}" ]]; then ARGS+=(--run "$RUN"); fi
if [[ -n "${SUMMARY:-}" ]]; then ARGS+=(--summary "$SUMMARY"); fi
exec "$PYTHON" -m A3C.utils.make_charts "${ARGS[@]}" "$@"
