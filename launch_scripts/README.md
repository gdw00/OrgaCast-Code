# OrgaCast launchers

Run from the repository root through `scripts/run.sh`. All training arguments after the Accelerate configuration path are forwarded to the trainer.

| Script | Purpose |
| --- | --- |
| `train_orgacast_historical47.sh` | Main 1000-update preset: full captions, selected sample split, compatible control initialization |
| `warmstart_orgacast.sh` | Shared training launcher; 300 updates by default, overridden by the main preset |
| `finetune_vae.sh` | Optional VAE training restricted to the saved training split |

## Main training

```bash
export ORGACAST_SPLIT_FILE="/path/to/your_split.json"
ORGACAST_OUTPUT_DIR=./outputs/orgacast_run1 \
  bash scripts/run.sh bash launch_scripts/train_orgacast_historical47.sh \
  launch_accelerate_configs/single_gpu.yaml
```

The original U-Net, VAE and CLIP stay frozen. ControlNet and new target-metadata adapters use learning rates of `5e-6`. The preset uses batch size 16, FP32, seed 42, validation at step 0 and every 100 updates, and a complete final checkpoint. Text is encoded in three CLIP chunks. It does not automatically run MC analysis or history-only forecasts.

GPU 0 trains; physical GPU 1 evaluates by default. Set `ORGACAST_EVALUATION_GPU` to select the worker GPU. Append `--evaluation_mode inline` for one GPU. Deterministic input tensors and target masks are cached in CPU memory. Read `evaluation_status.json` to confirm that queued evaluations finished.

Configure external components with `ORGACAST_BASE_MODEL`, `ORGACAST_UNET_PATH`, `ORGACAST_VAE_PATH` and `ORGACAST_LEGACY_CONTROLNET_PATH`. The initialization transfer is recorded in `initialization_transfer.json`. It starts fresh optimizer state; it is not a training-state resume. Launchers use local Hugging Face caches by default.

The loss weights are denoising MSE `1`, image MSE `1`, LPIPS `0.1` and hard-area error `1e-5`, with area surrogate temperature `0.02`. Masks and thresholds have no trainable parameters. These numeric choices are recorded implementation settings, not recovered paper hyperparameters.

## Planned budget and recovery

To start a 1500-update experiment:

```bash
ORGACAST_OUTPUT_DIR=./outputs/orgacast_run1500 \
  bash scripts/run.sh bash launch_scripts/train_orgacast_historical47.sh \
  launch_accelerate_configs/single_gpu.yaml \
  --max_train_steps=1500 --checkpointing_steps=1500 --checkpoint_preempt_steps=1500
```

For an interrupted run, repeat the original command and add `--resume_from_checkpoint latest`. Retain the original output directory, budget, data, evaluation mode and other training parameters. Initialization transfer is omitted during resume. Format-2 recovery checks core source hashes and restores both models, optimizer, scheduler, RNG, split and batch position. Changing a completed run's budget requires an explicit state migration, not ordinary resume. For checkpoints saved before source cleanup, use their matching archived source for exact recovery; model weights still load for prediction.

Private sample lists are not distributed. Supply a split through `ORGACAST_SPLIT_FILE` or `--split_file`; the command-line argument takes precedence. The JSON must contain nonempty, disjoint `train` and `validation` lists covering all complete sample IDs. Without an explicit split, an existing local `configs/splits/historical_47.json` remains the fallback. A new run with no available split stops before launching training; resume can use its saved checkpoint split. Matching membership does not establish identical preprocessing or initialization.

## Optional VAE training

```bash
bash scripts/run.sh bash launch_scripts/finetune_vae.sh \
  launch_accelerate_configs/single_gpu.yaml \
  --split_file "$ORGACAST_SPLIT_FILE" --report_to none
```

This entry trains on the selected training samples. It saves model weights and does not implement the forecasting trainer's full-state recovery. Its default checkpoint budget and learning rate are separate from the main model preset. Preparing a new VAE does not verify the training membership of an existing VAE asset.

## Prediction and evaluation

Use `inference.py` for ordinary prediction or labeled evaluation; `run_inference.sh` is a convenience wrapper around it. See the [root README](../README.md) for input layout and required model assets.

The trainer starts `tools/evaluate_queue.py` for asynchronous validation. It loads completed snapshots, evaluates the saved validation split and writes metrics, images and worker status. Use `--evaluation_mode inline` to evaluate within the training process instead.

Standalone uncertainty analysis, plotting and mask-comparison tools are maintained locally and excluded from this source release.
