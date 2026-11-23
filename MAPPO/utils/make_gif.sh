#!/usr/bin/env bash
# Generate a headless continuous-policy showcase with config-inferred weights.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-MAPPO/configs/multiwalker_tuned.yaml}"
OUT="${OUT:-MAPPO/assets/mappo_multiwalker.gif}"
EXTRA=()
[[ -n "${CHECKPOINT:-}" ]] && EXTRA+=(--checkpoint "$CHECKPOINT")
export SDL_VIDEODRIVER=dummy
exec "$PYTHON" -m MAPPO.utils.make_gif \
    --config "$CONFIG" --out "$OUT" "${EXTRA[@]}" "$@"
