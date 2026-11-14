#!/usr/bin/env bash
set -euo pipefail
ACCELERATE_CONFIG_FILE="${1:?Usage: finetune_vae.sh <accelerate-config> [arguments...]}"
shift
export HF_HOME="${HF_HOME:-./hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"
MODEL_NAME="${ORGACAST_BASE_MODEL:-stabilityai/stable-diffusion-2-1}"
DATA_DIR="${ORGACAST_DATA_DIR:-./data/cells}"
OUTPUT_DIR="${ORGACAST_VAE_OUTPUT_DIR:-./outputs/orgacast_vae_core}"
# Use the forecasting split's training samples only. Pass --split_file to reuse
# an existing experiment's exact sample IDs. This does not audit an old VAE.
accelerate launch --config_file "$ACCELERATE_CONFIG_FILE" finetune_vae.py \
  --pretrained_model_name_or_path="$MODEL_NAME" --train_data_dir="$DATA_DIR" \
  --output_dir="$OUTPUT_DIR" --resolution=256 --train_batch_size=8 \
  --max_train_steps=15000 --learning_rate=1e-5 --checkpointing_steps=5000 \
  --seed=42 --report_to=tensorboard "$@"
