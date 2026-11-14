#!/usr/bin/env bash
# Full captions and 1000 updates with a user-supplied or local private split.
set -euo pipefail
ACCELERATE_CONFIG_FILE="${1:?Usage: train_orgacast_historical47.sh <accelerate-config> [training arguments...]}"
shift
SPLIT_ARGS=()
EXPLICIT_SPLIT=0
RESUMING=0
for argument in "$@"; do
  case "$argument" in
    --split_file|--split_file=*) EXPLICIT_SPLIT=1 ;;
    --resume_from_checkpoint|--resume_from_checkpoint=*) RESUMING=1 ;;
  esac
done
if [[ "$EXPLICIT_SPLIT" == 0 ]]; then
  if [[ -n "${ORGACAST_SPLIT_FILE:-}" ]]; then
    SPLIT_ARGS=(--split_file="${ORGACAST_SPLIT_FILE}")
  elif [[ -f ./configs/splits/historical_47.json ]]; then
    SPLIT_ARGS=(--split_file=./configs/splits/historical_47.json)
  elif [[ "$RESUMING" == 0 ]]; then
    echo "Set ORGACAST_SPLIT_FILE or pass --split_file for your dataset; private sample lists are not distributed." >&2
    exit 2
  fi
fi
export ORGACAST_OUTPUT_DIR="${ORGACAST_OUTPUT_DIR:-./outputs/orgacast_historical47_chunked}"
bash launch_scripts/warmstart_orgacast.sh "${ACCELERATE_CONFIG_FILE}" \
  "${SPLIT_ARGS[@]}" \
  --text_encoding_mode=chunked --text_max_chunks=3 \
  --max_train_steps=1000 --checkpointing_steps=1000 --checkpoint_preempt_steps=1000 \
  --validation_steps=100 --report_to=none "$@"
