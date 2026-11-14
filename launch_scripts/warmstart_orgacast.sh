#!/usr/bin/env bash
# A short, auditable old-weight transfer experiment with the U-Net frozen.
set -euo pipefail
ACCELERATE_CONFIG_FILE="${1:?Usage: warmstart_orgacast.sh <accelerate-config> [training arguments...]}"
shift
export HF_HOME="${HF_HOME:-./hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"
MODEL_NAME="${ORGACAST_BASE_MODEL:-stabilityai/stable-diffusion-2-1}"
UNET_PATH="${ORGACAST_UNET_PATH:-./checkpoints/finetune_sd21_sn-satlas-fmow_snr5_md7norm_bs64/checkpoint-150000/unet}"
VAE_PATH="${ORGACAST_VAE_PATH:-./assets/legacy_outputs/finetuned-vae-on-cells/checkpoint-15000}"
CONTROL_PATH="${ORGACAST_LEGACY_CONTROLNET_PATH:-./assets/legacy_outputs/6_controlnet3d-mixattn_sd21_md7norm_fmow_temporal/checkpoint-1250/controlnet}"
OUT_DIR="${ORGACAST_OUTPUT_DIR:-./outputs/orgacast_warmstart_pilot}"
INITIALIZATION_ARGS=(--legacy_controlnet_init="${CONTROL_PATH}")
for argument in "$@"; do
  case "$argument" in
    --resume_from_checkpoint|--resume_from_checkpoint=*) INITIALIZATION_ARGS=() ;;
  esac
done
accelerate launch --config_file="${ACCELERATE_CONFIG_FILE}" --mixed_precision=no --main_process_port=45143 train_controlnet.py \
  --pretrained_model_name_or_path="${MODEL_NAME}" --unet_path="${UNET_PATH}" --vae_path="${VAE_PATH}" \
  "${INITIALIZATION_ARGS[@]}" --dataset=cell_image_temporal --train_data_dir=./data/cells \
  --num_metadata=4 --num_cond=3 --num_channels=3 --input_days day7 day9 day11 \
  --temporal --temporal_attn --temporal_kernel_size=3 --temporal_dropout=0.1 --metadata_dropout=0 \
  --resolution=256 --train_batch_size=16 --gradient_accumulation_steps=1 --gradient_checkpointing \
  --cache_preprocessed_data --cache_target_masks --evaluation_mode=async --evaluation_gpu="${ORGACAST_EVALUATION_GPU:-1}" \
  --max_train_steps=300 --checkpointing_steps=300 --checkpoint_preempt_steps=300 \
  --learning_rate=5e-6 --metadata_learning_rate=5e-6 --max_grad_norm=1 \
  --image_mse_weight=1 --lpips_loss_weight=0.1 --diffusion_loss_weight=1 \
  --morph_loss_lambda=0.00001 --morph_temperature=0.02 \
  --lr_scheduler=constant --lr_warmup_steps=0 --seed=42 --drop_temporal_cond=0 \
  --validate_initialization --validation_steps=50 --num_inference_steps=50 \
  --skip_final_pipeline_export --output_dir="${OUT_DIR}" "$@"
