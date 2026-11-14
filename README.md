# OrgaCast

OrgaCast forecasts future fluorescence organoid images from an observed image sequence, experimental metadata and text descriptions. The current configuration uses Days 7, 9 and 11 to predict Day 15.

The model combines a pretrained latent diffusion U-Net with a temporal ControlNet. Parallel temporal convolution and attention integrate image history, while separate metadata encoders represent observed and target conditions. The original U-Net, VAE and text encoder stay frozen; the control branch and new metadata adapters are trained. Repeated stochastic forecasts provide ensemble predictions and pixel-wise variance maps.

## Installation

The tested environment uses Python 3.10 and PyTorch 2.2.2 with CUDA 12.1.

```bash
conda create -n orgacast python=3.10
conda activate orgacast
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
pip install -e ./vendor/diffusers --no-deps
pip install -e .
```

Run commands from the repository root using `scripts/run.sh` to select the bundled Diffusers runtime.

## Training

Configure the model components and link the image sequences:

```bash
export ORGACAST_BASE_MODEL="stabilityai/stable-diffusion-2-1"
export ORGACAST_UNET_PATH="/path/to/pretrained_unet"
export ORGACAST_VAE_PATH="/path/to/cell_vae"
export ORGACAST_LEGACY_CONTROLNET_PATH="/path/to/legacy_controlnet"
export HF_HOME="$PWD/hf_cache"
python scripts/link_local_assets.py --cells /path/to/cell_sequences
```

Each sequence contains CH1, CH2 and CH3 TIFF images for the observed days and, for training, the target day.

Launch training:

```bash
export ORGACAST_OUTPUT_DIR="./outputs/orgacast_run1"
export ORGACAST_SPLIT_FILE="/path/to/your_split.json"
bash scripts/run.sh bash launch_scripts/train_orgacast_historical47.sh \
  launch_accelerate_configs/single_gpu.yaml
```

Training uses GPU 0 and asynchronous validation uses GPU 1. For one GPU, append `--evaluation_mode inline`. Checkpoints, metrics and images are written to `ORGACAST_OUTPUT_DIR`. See the [launcher guide](launch_scripts/README.md) for training options, checkpoint recovery and optional VAE training.

## Prediction

```bash
export ORGACAST_CHECKPOINT="./outputs/orgacast_run1/checkpoint-1000"
bash scripts/run.sh python inference.py \
  --checkpoint "$ORGACAST_CHECKPOINT" \
  --data_dir ./data/cells --with_targets --subset validation \
  --inference_batch_size 13 --num_inference_steps 50 --seed 42 \
  --output_dir ./outputs/orgacast_predictions
```

This evaluates the checkpoint's saved validation cases and writes predictions, previews and metrics. For history-only forecasting, omit `--with_targets --subset validation` and point `--data_dir` to the observed images.

## License

Code is distributed under [Apache 2.0](LICENSE). Bundled [Hugging Face Diffusers](vendor/diffusers) retains its [license](vendor/diffusers/LICENSE) and source copyright notices.
