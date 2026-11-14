#!/usr/bin/env bash
set -euo pipefail
# Use a complete checkpoint from the NEW method; no historical mixed components.
CHECKPOINT_DIR="${1:?Usage: run_inference.sh <format-2-checkpoint> [inference arguments...]}"
shift
export HF_HOME="${HF_HOME:-./hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"
exec python inference.py --checkpoint "$CHECKPOINT_DIR" \
  --data_dir="${ORGACAST_INFERENCE_DATA:-./data/prediction_for_day_15}" \
  --output_dir="${ORGACAST_INFERENCE_OUTPUT:-./outputs/orgacast_predictions}" "$@"
