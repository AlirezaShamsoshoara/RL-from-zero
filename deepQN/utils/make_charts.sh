#!/usr/bin/env bash
# Require an explicit DQN training run rather than guessing the newest W&B file.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
PYTHON="${PYTHON:-python}"
OUT_DIR="${OUT_DIR:-deepQN/assets}"
TITLE="${TITLE:-DQN MountainCar-v0}"
ENV_ID="${ENV_ID:-MountainCar-v0}"
EXTRA=()
[[ -n "${RUN:-}" ]] && EXTRA+=(--run "$RUN")
[[ -n "${SUMMARY:-}" ]] && EXTRA+=(--summary "$SUMMARY")
exec "$PYTHON" -m deepQN.utils.make_charts \
    --out-dir "$OUT_DIR" --title "$TITLE" --env-id "$ENV_ID" "${EXTRA[@]}" "$@"
