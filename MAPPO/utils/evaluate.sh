#!/usr/bin/env bash
# Use the activated Python and infer weights from the selected evaluation config.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-MAPPO/configs/multiwalker_tuned.yaml}"
OUT="${OUT:-MAPPO/logs/evaluation.json}"
EXTRA=()
[[ -n "${CHECKPOINT:-}" ]] && EXTRA+=(--checkpoint "$CHECKPOINT")
export WANDB_MODE=offline
exec "$PYTHON" -m MAPPO.utils.evaluate \
    --config "$CONFIG" --out "$OUT" "${EXTRA[@]}" "$@"
