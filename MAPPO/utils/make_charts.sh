#!/usr/bin/env bash
# Choose a MAPPO training datastore explicitly, not the newest unrelated W&B run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-MAPPO/assets}"
TITLE="${TITLE:-MAPPO MultiWalker}"
EXTRA=()
[[ -n "${RUN:-}" ]] && EXTRA+=(--run "$RUN")
[[ -n "${SUMMARY:-}" ]] && EXTRA+=(--summary "$SUMMARY")
exec "$PYTHON" -m MAPPO.utils.make_charts \
    --out-dir "$OUT_DIR" --title "$TITLE" "${EXTRA[@]}" "$@"
