###########################################################################
# References:
# https://github.com/huggingface/diffusers/
###########################################################################

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

from pathlib import Path
import argparse
import json
import logging
import math
import os
import random
import re
import shutil
import time

import diffusers
import einops
import lpips
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.utils as vutils
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UniPCMultistepScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import create_repo, upload_folder
from packaging import version
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from orgacast.orgacast_unet import OrgaCastUNet
from orgacast.controlnet_3d import ControlNetModel3D
from orgacast.pipeline_controlnet import StableDiffusionControlNetPipeline
from orgacast.cell_data import MultiConditionTimeSeriesDataset, InferenceOnlyDataset, collate_cells
from orgacast.data_cache import CachedCellDataset
from orgacast.evaluation_queue import AsyncEvaluationQueue
from orgacast.paper_training import (
    METHOD_VERSION, METADATA_VERSION, initialize_cell_metadata, set_training_policy,
    restore_training_modes, target_metadata_kwargs, trainable_parameters, control_scale,
)
from orgacast.forecast import forecast_batch
from orgacast.text_conditioning import encode_text_conditioning, caption_audit
from orgacast.evaluation import image_metrics, save_metric_records
from orgacast.image_losses import predicted_clean_latents, image_objectives, hard_foreground_area_loss
from orgacast.masking import foreground_masks, mask_configuration
from orgacast.uncertainty import mc_dropout, ensemble_statistics
from orgacast.warmstart import initialize_from_legacy_controlnet
from orgacast.training_state import (
    StopAtUpdateBoundary, accumulate_joint_models, assert_optimizer_binding,
    cell_dataset_split, check_resume_config, read_training_checkpoint,
    register_joint_model_hooks, resolve_resume_checkpoint, save_training_checkpoint,
    training_run_config, write_json,
)

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def generate_with_uncertainty(
    vae, text_encoder, unet, controlnet, noise_scheduler,
    accelerator, weight_dtype, args, data_batch, output_dir: Path, num_mc_samples: int = 20,
):
    """Fixed initial noise, stochastic trained dropout, and paper population variance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = accelerator.device
    unet, controlnet = accelerator.unwrap_model(unet), accelerator.unwrap_model(controlnet)
    vae.eval()
    text_encoder.eval()
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices), mc_dropout(unet, controlnet) as dropout_layers:
        torch.manual_seed(args.seed)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        hidden = encode_text_conditioning(text_encoder, data_batch["input_ids"]).to(dtype=weight_dtype)
        images = data_batch["conditioning_pixel_values"].to(device=device, dtype=weight_dtype)
        images = einops.rearrange(images, "b t c h w -> b c t h w")
        metadata = data_batch["metadata"].to(device=device, dtype=weight_dtype)
        history = data_batch["conditioning_metadata"].to(device=device, dtype=weight_dtype)
        if args.controlnet_scales is not None:
            scales = torch.tensor(args.controlnet_scales, device=device, dtype=weight_dtype).view(1, 1, -1, 1, 1)
            images = images * scales
        initial_noise = torch.randn(
            (images.shape[0], unet.config.in_channels, args.resolution // 8, args.resolution // 8),
            generator=generator, device=device, dtype=weight_dtype,
        )
        inference_scheduler = UniPCMultistepScheduler.from_config(noise_scheduler.config)
        routed_metadata = target_metadata_kwargs(controlnet, unet, metadata)
        predictions = []
        for _ in tqdm(range(num_mc_samples), desc="MC Dropout", leave=False):
            latents = initial_noise.clone()
            inference_scheduler.set_timesteps(args.num_inference_steps)
            for timestep in inference_scheduler.timesteps:
                residuals, middle = controlnet(
                    latents, timestep, encoder_hidden_states=hidden,
                    controlnet_cond=images, conditioning_scale=control_scale(args),
                    metadata=metadata, cond_metadata=history, return_dict=False,
                    **routed_metadata,
                )
                prediction = unet(
                    latents, timestep, encoder_hidden_states=hidden, metadata=metadata,
                    down_block_additional_residuals=[value.to(weight_dtype) for value in residuals],
                    mid_block_additional_residual=middle.to(weight_dtype),
                ).sample
                latents = inference_scheduler.step(prediction, timestep, latents).prev_sample
            decoded = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
            predictions.append((decoded / 2 + 0.5).clamp(0, 1).float())
        mean, variance = ensemble_statistics(torch.stack(predictions))
        write_json(output_dir / "uncertainty_config.json", {
            "method": "mc_dropout_fixed_noise", "samples": num_mc_samples,
            "variance_denominator": "R", "image_range": [0, 1],
            "tensor_channels": ["CH1", "CH2", "CH3"], "display_channels": ["CH2", "CH1", "CH3"],
            "dropout_layers": dropout_layers, "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "method_version": METHOD_VERSION,
        })
        for index, key in enumerate(data_batch["sample_keys"]):
            np.savez_compressed(
                output_dir / f"{key}_uncertainty.npz",
                mean=mean[index].cpu().numpy(), variance=variance[index].cpu().numpy(),
            )
            vutils.save_image(mean[index][[1, 0, 2]], output_dir / f"{key}_mean_prediction.png")
            display = variance[index].mean(dim=0)
            display = (display - display.min()) / (display.max() - display.min() + 1e-6)
            colored = plt.get_cmap("viridis")(display.cpu().numpy())[..., :3]
            vutils.save_image(torch.from_numpy(colored).permute(2, 0, 1), output_dir / f"{key}_uncertainty_map.png")
            if "pixel_values" in data_batch:
                target = (data_batch["pixel_values"][index] / 2 + 0.5).clamp(0, 1)
                vutils.save_image(target[[1, 0, 2]], output_dir / f"{key}_ground_truth.png")
    logger.info(f"Saved MC Dropout mean, population variance and display maps to {output_dir}")


# def log_validation(vae, text_encoder, tokenizer, unet, controlnet, noise_scheduler, args, accelerator, weight_dtype, step, val_dataloader, loss_fn_lpips):
def log_validation(vae, text_encoder, tokenizer, unet, controlnet, noise_scheduler,
                   args, accelerator, weight_dtype, step, val_dataloader,
                   loss_fn_lpips, inference_dataloader=None):
    """Use the same forecast/preprocessing path for labeled and unlabeled cells."""
    unet, controlnet = accelerator.unwrap_model(unet), accelerator.unwrap_model(controlnet)
    for model in (unet, controlnet, vae, text_encoder):
        model.eval()
    output = Path(args.output_dir)
    validation_dir = output / "validation_samples" / f"gs_{step}"
    validation_dir.mkdir(parents=True, exist_ok=True)
    write_json(validation_dir / "mask_config.json", mask_configuration())
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    records = []
    try:
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validating"):
                predictions = forecast_batch(unet, controlnet, vae, text_encoder, noise_scheduler, batch, args, generator)
                targets = (batch["pixel_values"].to(predictions.device) / 2 + 0.5).clamp(0, 1)
                predicted_masks = foreground_masks(predictions)
                target_masks = (batch["target_foreground_mask"].to(device=predictions.device, dtype=torch.float32)
                                if "target_foreground_mask" in batch else foreground_masks(targets))
                for index, key in enumerate(batch["sample_keys"]):
                    records.append(dict(sample_key=key, **image_metrics(predictions[index], targets[index], loss_fn_lpips)))
                    vutils.save_image(predictions[index][[1, 0, 2]], validation_dir / f"{key}_predicted.png")
                    vutils.save_image(targets[index][[1, 0, 2]], validation_dir / f"{key}_target.png")
                    vutils.save_image(predicted_masks[index], validation_dir / f"{key}_predicted_mask.png")
                    vutils.save_image(target_masks[index], validation_dir / f"{key}_target_mask.png")
                    conditions = (batch["conditioning_pixel_values"][index] / 2 + 0.5).clamp(0, 1)
                    vutils.save_image(conditions[:, [1, 0, 2]], validation_dir / f"{key}_conditions_grid.png", nrow=len(args.input_days))
        metrics_dir = output / "validation_metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics = save_metric_records(metrics_dir / f"gs_{step}.json", records, step=step)
        if metrics:
            accelerator.log({f"Validation/{name.upper()}": value for name, value in metrics.items()}, step=step)
            with (output / "validation_metrics.txt").open("a") as stream:
                stream.write(f"Step: {step}, SSIM: {metrics['ssim']:.4f}, PSNR: {metrics['psnr']:.4f}, LPIPS: {metrics['lpips']:.4f}\n")
            logger.info(f"Validation at step {step}: {metrics}")
        if inference_dataloader is not None:
            inference_dir = output / "inference_only_samples" / f"gs_{step}"
            inference_dir.mkdir(parents=True, exist_ok=True)
            write_json(inference_dir / "mask_config.json", mask_configuration())
            generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            for batch in tqdm(inference_dataloader, desc="Inference Only"):
                predictions = forecast_batch(unet, controlnet, vae, text_encoder, noise_scheduler, batch, args, generator)
                predicted_masks = foreground_masks(predictions)
                for index, key in enumerate(batch["sample_keys"]):
                    vutils.save_image(predictions[index].amax(dim=0, keepdim=True), inference_dir / f"{key}_predicted_MAX_OVERLAY.png")
                    vutils.save_image(predictions[index][[1, 0, 2]], inference_dir / f"{key}_predicted_RGB_swapped.png")
                    vutils.save_image(predicted_masks[index], inference_dir / f"{key}_predicted_mask.png")
                    conditions = (batch["conditioning_pixel_values"][index] / 2 + 0.5).clamp(0, 1)
                    vutils.save_image(conditions[:, [1, 0, 2]], inference_dir / f"{key}_conditions_grid.png", nrow=len(args.input_days))
        return metrics
    finally:
        restore_training_modes(unet, controlnet, vae, text_encoder)


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Train OrgaCast on fluorescence organoid time series.")
    parser.add_argument("--legacy_controlnet_init", help="Audited partial transfer from a local seven-field legacy ControlNet into a NEW parallel control branch.")
    parser.add_argument("--validate_initialization", action="store_true", help="Save validation predictions at step 0 before the first optimizer update.")
    parser.add_argument("--skip_final_pipeline_export", action="store_true", help="Keep the complete final checkpoint without additional duplicate pipeline exports.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Initialize a new training stage from these ControlNet weights, with a fresh optimizer and step 0."
        " If not specified, initialize ControlNet from the U-Net. Mutually exclusive with --resume_from_checkpoint.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help=(
            "Revision of pretrained model identifier from huggingface.co/models. Trainable model components should be"
            " float32 precision."
        ),
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/orgacast",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for the saved cell split and training randomness.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--inference_data_dir",
        type=str,
        default=None,
        help="A folder containing the inference-only data (without target images)."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoint_preempt_steps",
        type=int,
        default=None,
        help=(
            "Save a recent checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Resume the same run from a complete OrgaCast joint checkpoint, restoring both models and training state. "
            "Use 'latest' for the newest complete checkpoint in --output_dir. Legacy weights are initialization only."
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "Root folder containing condition/day directories with CH1, CH2 and CH3 TIFF images."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="orgacast",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )

    # CUSTOM: Added for OrgaCast
    parser.add_argument("--wandb", type=str, default=None, help="Name of wandb project. None to not use wandb.")
    parser.add_argument("--dataset", type=str, default='cell_image_temporal', choices=['cell_image_temporal'])
    parser.add_argument(
        "--unet_path", type=str, default=None,
    )
    parser.add_argument("--cond_resolution", type=int, default=None,
                        help="Whether to set different resolution for cond pixels vs output image.")
    parser.add_argument(
        "--num_metadata", type=int, default=4,
    )
    parser.add_argument(
        "--num_cond", type=int, default=3, help="Number of conditioning images to cat in channel dim."
    )
    parser.add_argument(
        "--drop_cond", type=float, default=0.0, help="Percentage of time to drop a conditioning image."
    )
    parser.add_argument(
        "--drop_temporal_cond", type=float, default=0.0,
        help="Probability of dropping some temporal conditioning frames during training. "
             "This helps the model learn to predict from incomplete time series (e.g., only first day). "
             "Range: 0.0 (no dropout) to 1.0 (always dropout some frames)."
    )
    parser.add_argument(
        "--test_second_day_only", action="store_true", default=False,
        help="During validation, only use the second day image (day9) as input to test the model's ability "
             "to predict future states from minimal temporal information."
    )
    parser.add_argument(
        "--num_channels", type=int, default=3,
    )
    parser.add_argument("--temporal", action="store_true", default=True, help="Use 3D controlnet")
    parser.add_argument("--temporal_attn", action="store_true", default=True, help="Use temporal attention for temporal models.")
    parser.add_argument("--multi", action="store_true", default=False, help="Train multiple controlnets, one for each cond image")
    parser.add_argument(
        "--morph_loss_lambda", type=float, default=0.00001,
        help="Weight for pixel-count hard area error with a sigmoid surrogate gradient."
    )
    parser.add_argument(
        "--morph_temperature", type=float, default=0.02,
        help="Fixed surrogate sigmoid temperature in channel-mean [0,1] intensity units."
    )
    parser.add_argument(
        "--run_uncertainty_analysis",
        action="store_true",
        help="If set, run uncertainty analysis on the validation set after training."
    )
    parser.add_argument(
        "--uncertainty_output_dir",
        type=str,
        default="uncertainty_results",
        help="Directory to save uncertainty analysis results."
    )
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=20,
        help="Number of Monte Carlo samples for uncertainty estimation."
    )

    parser.add_argument(
            "--controlnet_scales",
            type=float,
            nargs="+",
            default=None,
            help="A list of weights for each controlnet in MultiControlNet, implementing the proximity principle. e.g., --controlnet_scales 0.8 0.9 1.0"
        )


    parser.add_argument("--input_days", nargs="+", choices=["day7", "day9", "day11"], default=["day7", "day9", "day11"])
    parser.add_argument("--conditioning_ablation", choices=["none", "text", "metadata", "images"], default="none")
    parser.add_argument("--metadata_dropout", type=float, default=0.0)
    parser.add_argument("--temporal_dropout", type=float, default=0.1, help="Implementation choice; the paper does not report its value.")
    parser.add_argument("--temporal_kernel_size", type=int, default=3)
    parser.add_argument("--image_mse_weight", type=float, default=1.0)
    parser.add_argument("--lpips_loss_weight", type=float, default=0.1, help="Configurable lambda_p; 0.1 is a provisional implementation choice, not a recovered paper value.")
    parser.add_argument("--diffusion_loss_weight", type=float, default=1.0, help="Denoising MSE weight; 1.0 is the selected engineering preset, not a confirmed paper value. Set 0 for an image-objective-only ablation.")
    parser.add_argument("--metadata_learning_rate", type=float, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--split_file", type=str, default=None, help="Reuse a saved sample split for controlled comparisons.")
    parser.add_argument("--text_encoding_mode", choices=["truncate", "chunked"], default="chunked",
                        help="Encode full captions as CLIP chunks; truncate reproduces the former text path.")
    parser.add_argument("--text_max_chunks", type=int, default=3,
                        help="Maximum fixed CLIP chunks; overflow raises an error instead of dropping text.")
    parser.add_argument("--vae_path", type=str, default="./assets/legacy_outputs/finetuned-vae-on-cells/checkpoint-15000")
    parser.add_argument("--cache_preprocessed_data", action="store_true", help="Cache the exact deterministic cell tensors in CPU memory.")
    parser.add_argument("--cache_target_masks", action="store_true", help="Cache fixed target masks with the preprocessed tensors.")
    parser.add_argument("--evaluation_mode", choices=["inline", "async"], default="inline")
    parser.add_argument("--evaluation_gpu", default="1", help="Physical GPU index for the independent evaluation worker.")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.test_second_day_only or args.drop_cond or args.drop_temporal_cond:
        parser.error("Use --input_days/--num_cond for paper temporal ablations; legacy condition masking is not part of this preset.")
    if args.num_metadata != 4 or args.num_channels != 3 or not args.temporal or args.multi:
        parser.error("The paper-aligned entry uses four metadata fields, three channels and one temporal ControlNet.")
    if not args.input_days or len(set(args.input_days)) != len(args.input_days) or args.input_days != sorted(args.input_days, key=lambda day: int(day[3:])):
        parser.error("--input_days must be unique and chronological.")
    if args.num_cond != len(args.input_days):
        parser.error("--num_cond must equal the number of --input_days.")
    if args.temporal_kernel_size < 3 or args.temporal_kernel_size % 2 != 1:
        parser.error("--temporal_kernel_size must be odd and at least 3.")
    if not 0 <= args.temporal_dropout < 1 or not 0 <= args.metadata_dropout < 1:
        parser.error("Dropout probabilities must be in [0, 1).")
    if args.run_uncertainty_analysis and (args.mc_samples < 2 or args.temporal_dropout <= 0):
        parser.error("MC Dropout requires at least two samples and positive trained temporal dropout.")
    loss_weights = (args.image_mse_weight, args.lpips_loss_weight, args.diffusion_loss_weight, args.morph_loss_lambda)
    if any(not math.isfinite(weight) or weight < 0 for weight in loss_weights):
        parser.error("Loss weights must be finite and nonnegative.")
    if sum(loss_weights) <= 0:
        parser.error("Enable at least one training objective.")
    if not math.isfinite(args.morph_temperature) or args.morph_temperature <= 0:
        parser.error("--morph_temperature must be finite and positive.")
    if args.num_inference_steps < 1:
        parser.error("--num_inference_steps must be positive.")
    if args.text_max_chunks < 1:
        parser.error("--text_max_chunks must be positive.")
    if args.controlnet_scales is not None and len(args.controlnet_scales) != len(args.input_days):
        parser.error("One control scale is required for each observed day.")
    if args.cache_target_masks and not args.cache_preprocessed_data:
        parser.error("--cache_target_masks requires --cache_preprocessed_data.")
    if args.evaluation_mode == "async" and not args.evaluation_gpu.isdigit():
        parser.error("--evaluation_gpu must be a single physical GPU index.")
    if args.metadata_learning_rate is None:
        args.metadata_learning_rate = args.learning_rate

    if args.resume_from_checkpoint and args.controlnet_model_name_or_path:
        parser.error("Use either --controlnet_model_name_or_path for a new stage or --resume_from_checkpoint for continuation.")
    if args.legacy_controlnet_init and (args.resume_from_checkpoint or args.controlnet_model_name_or_path):
        parser.error("--legacy_controlnet_init starts a new stage and cannot be combined with another initialization or resume.")
    if args.train_batch_size < 1 or args.gradient_accumulation_steps < 1:
        parser.error("Batch size and gradient accumulation must be positive.")
    if args.checkpointing_steps < 1 or (args.checkpoint_preempt_steps is not None and args.checkpoint_preempt_steps < 1):
        parser.error("Checkpoint intervals must be positive.")
    if args.max_train_steps is not None and args.max_train_steps < 1:
        parser.error("--max_train_steps must be positive.")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution < 16 or args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be at least 16 and divisible by 8 for the VAE, ControlNet and validation metrics."
        )

    if args.unet_path is None:
        args.unet_path = args.pretrained_model_name_or_path
    print(f"Using unet from {args.unet_path}")
    if args.cond_resolution is None:
        args.cond_resolution = args.resolution
    if args.cond_resolution != args.resolution:
        parser.error("The shared cell preprocessing uses the same resolution for history and target.")
    return args


def main(args):
    # Resolve and validate resume files before allocating models or downloading anything.
    resume_path = resolve_resume_checkpoint(args.resume_from_checkpoint, args.output_dir)
    saved_state, saved_split = read_training_checkpoint(resume_path) if resume_path else (None, None)
    output_dir = Path(args.output_dir)
    if resume_path is None and (
        (output_dir / "training_metrics.txt").exists() or any(output_dir.glob("checkpoint-*"))
    ):
        raise ValueError("This output directory already contains a training run; resume it or choose a new --output_dir.")
    if saved_state:
        for checkpoint in output_dir.glob("checkpoint-*"):
            match = re.fullmatch(r"checkpoint-(\d+)", checkpoint.name)
            if match and int(match.group(1)) > saved_state["global_step"] and (checkpoint / "checkpoint_complete.json").is_file():
                raise ValueError("The output directory contains later checkpoints; use a new --output_dir to branch from an older checkpoint.")
    if args.controlnet_model_name_or_path:
        source = args.controlnet_model_name_or_path
        if (source.startswith((".", "/")) or len(Path(source).parts) > 2) and not Path(source).is_dir():
            raise FileNotFoundError(f"ControlNet initialization directory does not exist: {source}")
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=None if args.report_to == "none" else args.report_to,
        project_config=accelerator_project_config,
    )
    if accelerator.distributed_type.name not in {"NO", "MULTI_CPU", "MULTI_GPU"}:
        raise ValueError("Joint OrgaCast checkpoints support single-process and DDP training; this backend needs separate checkpoint handling.")
    if args.evaluation_mode == "async":
        if accelerator.num_processes != 1 or accelerator.device.type != "cuda" or accelerator.mixed_precision != "no":
            raise ValueError("Async evaluation currently supports one FP32 training process and a separate CUDA GPU.")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        training_gpu = visible.split(",")[0] if visible else "0"
        if args.evaluation_gpu == training_gpu:
            raise ValueError("Training and asynchronous evaluation must use different GPUs.")
    if saved_state:
        early_fields = ("seed", "temporal", "temporal_attn", "multi", "num_metadata", "num_cond", "num_channels", "resolution", "cond_resolution")
        check_resume_config(
            {key: saved_state["run_config"][key] for key in early_fields},
            {key: getattr(args, key) for key in early_fields},
        )
    logger.info("Initializing LPIPS loss function...")

    loss_fn_lpips = lpips.LPIPS(net='vgg').to(accelerator.device)

    loss_fn_lpips.eval()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.wandb is not None:
            wandb.init(project=args.wandb)
            wandb.config.update(args)

        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # import correct text encoder class
    text_encoder_cls = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, args.revision)

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder = text_encoder_cls.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
    )
    # vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    finetuned_vae_path = args.vae_path
    logger.info(f"Loading fixed VAE from: {finetuned_vae_path}")
    vae = AutoencoderKL.from_pretrained(finetuned_vae_path)
    unet_source = str(resume_path / "unet") if resume_path else args.unet_path
    unet = OrgaCastUNet.from_pretrained(unet_source, subfolder="unet", low_cpu_mem_usage=True)
    if resume_path:
        if getattr(unet.config, "metadata_semantics", None) != METADATA_VERSION:
            raise ValueError("Checkpoint uses a different metadata representation.")
    else:
        initialize_cell_metadata(unet)

    transfer_report = None
    controlnet_source = str(resume_path / "controlnet") if resume_path else args.controlnet_model_name_or_path
    if controlnet_source:
        config = ControlNetModel3D.load_config(controlnet_source)
        expected = {"temporal_fusion_mode": "parallel", "metadata_conditioning": "independent",
                    "num_metadata": 4, "temporal_dropout": args.temporal_dropout,
                    "temporal_kernel_size": args.temporal_kernel_size}
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError("ControlNet weights do not match the paper-aligned architecture. Initialize the new control branch from the U-Net, or supply a matching new checkpoint.")
        controlnet = ControlNetModel3D.from_pretrained(controlnet_source)
    else:
        controlnet = ControlNetModel3D.from_unet(
            unet, conditioning_in_channels=3, use_temporal_transformer=True,
            temporal_fusion_mode="parallel", metadata_conditioning="independent",
            temporal_kernel_size=args.temporal_kernel_size, temporal_dropout=args.temporal_dropout,
        )
        if args.legacy_controlnet_init:
            transfer_report = initialize_from_legacy_controlnet(controlnet, args.legacy_controlnet_init, len(args.input_days))
            if accelerator.is_main_process:
                write_json(output_dir / "initialization_transfer.json", transfer_report)
            logger.info(f"Reused {transfer_report['reused_source_fraction']:.1%} of legacy parameter values; new metadata and temporal attention remain newly initialized.")
    set_training_policy(unet, controlnet, vae, text_encoder, loss_fn_lpips)
    if accelerator.is_main_process:
        write_json(output_dir / "training_policy.json", {
            "unet_backbone_trainable": any(p.requires_grad for n, p in unet.named_parameters() if not n.startswith("metadata_embedding.")),
            "unet_trainable_parameters": [n for n, p in unet.named_parameters() if p.requires_grad],
            "controlnet_trainable_parameter_values": sum(p.numel() for p in controlnet.parameters() if p.requires_grad),
            "vae_trainable": any(p.requires_grad for p in vae.parameters()),
            "text_encoder_trainable": any(p.requires_grad for p in text_encoder.parameters()),
            "controlnet_learning_rate": args.learning_rate,
            "metadata_learning_rate": args.metadata_learning_rate,
        })
    logger.info("Training the control branch and four-field target metadata adapters; pretrained U-Net, VAE and CLIP are frozen.")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if accelerator.unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {accelerator.unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        lr_scale = args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        args.learning_rate *= lr_scale
        args.metadata_learning_rate *= lr_scale

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    params_to_optimize = [
        {"params": list(trainable_parameters(controlnet)), "lr": args.learning_rate},
        {"params": list(trainable_parameters(unet)), "lr": args.metadata_learning_rate},
    ]
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    full_dataset = MultiConditionTimeSeriesDataset(
        root_dir=args.train_data_dir, transform=None, tokenizer=tokenizer, args=args
    )
    if hasattr(full_dataset, "caption_texts"):
        text_audit = caption_audit(tokenizer, args, full_dataset.caption_texts)
        if accelerator.is_main_process:
            write_json(output_dir / "text_conditioning_audit.json", text_audit)
        logger.info(f"Text encoding: {text_audit['version']}; {text_audit['max_chunks']} chunk(s); "
                    f"{sum(row['truncated'] for row in text_audit['records'])} truncated captions")
    if args.cache_preprocessed_data:
        full_dataset = CachedCellDataset(full_dataset, cache_target_masks=args.cache_target_masks)
        logger.info(f"Cached {len(full_dataset)} cell sequences and fixed target masks: {full_dataset.cache_bytes / 2**20:.1f} MiB")
    train_dataset, val_dataset, dataset_split = cell_dataset_split(
        full_dataset, args.seed, saved=(saved_split if resume_path else json.loads(Path(args.split_file).read_text()) if args.split_file else None)
    )
    logger.info(f"Training set size: {len(train_dataset)}; validation set size: {len(val_dataset)}")


    #     return dict_of_batches

    # A separate generator prevents iterator creation on resume from advancing
    # the model/noise RNG. Keep the existing sequential order within each split.
    train_loader_generator = torch.Generator().manual_seed(args.seed)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        generator=train_loader_generator,
        collate_fn=collate_cells,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        generator=torch.Generator().manual_seed(args.seed + 1),
        shuffle=False,
        collate_fn=collate_cells,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    inference_dataloader = None
    if args.inference_data_dir:
        logger.info(f"Loading inference-only dataset from {args.inference_data_dir}")
        inference_dataset = InferenceOnlyDataset(
            root_dir=args.inference_data_dir,
            transform=None,
            tokenizer=tokenizer,
            args=args
        )

        logger.info(f"Inference-only dataset found {len(inference_dataset)} samples.")

        inference_dataloader = torch.utils.data.DataLoader(
            inference_dataset,
            generator=torch.Generator().manual_seed(args.seed + 2),
            shuffle=False,
            collate_fn=collate_cells,
            batch_size=args.train_batch_size,
            num_workers=args.dataloader_num_workers,
        )
        logger.info(f"Inference-only dataset size: {len(inference_dataset)}")
    num_samples = len(train_dataset)
    num_batches = math.ceil(num_samples / args.train_batch_size)


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(num_batches / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    assert_optimizer_binding(optimizer, unet, controlnet)
    unet, controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, controlnet, optimizer, train_dataloader, lr_scheduler
    )
    assert_optimizer_binding(optimizer, unet, controlnet)
    register_joint_model_hooks(accelerator, unet, controlnet)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Accelerate manages both trainable models and their autocast wrappers.
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = {
            key: value if isinstance(value, (str, int, float, bool)) else json.dumps(value)
            for key, value in vars(args).items() if key != "hub_token"
        }

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num training examples = {len(train_dataset)}")
    logger.info(f"  Num batches per process each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    run_config = training_run_config(args, accelerator)
    inference_samples = list(inference_dataset.samples) if inference_dataloader is not None else []
    if saved_split is not None and saved_split.get("inference", []) != inference_samples:
        raise ValueError("Inference sample IDs differ from the saved run.")
    dataset_split["inference"] = inference_samples
    if saved_state:
        check_resume_config(saved_state["run_config"], run_config)
        training_state = saved_state
    else:
        training_state = {
            "global_step": 0, "epoch": 0, "next_batch": 0,
            "run_config": run_config,
            "initialization": {
                "unet": args.unet_path,
                "controlnet": args.legacy_controlnet_init or args.controlnet_model_name_or_path or "initialized_from_unet",
                "vae": finetuned_vae_path,
            },
        }
        if transfer_report is not None:
            training_state["initialization"]["legacy_transfer"] = transfer_report
    global_step = training_state["global_step"]
    first_epoch, first_batch = training_state["epoch"], training_state["next_batch"]
    expected_step = first_epoch * num_update_steps_per_epoch + first_batch // args.gradient_accumulation_steps
    if first_batch >= len(train_dataloader) or first_batch % args.gradient_accumulation_steps != 0 or global_step != expected_step:
        raise ValueError("Saved epoch/batch position is incompatible with the current dataloader.")
    if global_step > args.max_train_steps:
        raise ValueError("Checkpoint progress exceeds --max_train_steps.")
    if accelerator.is_main_process:
        write_json(output_dir / "dataset_split.json", dataset_split)
        write_json(output_dir / "run_config.json", run_config)
    if resume_path:
        # The hooks copy model tensors in-place. Optimizer/scheduler/RNG states
        # are restored into the objects already registered with Accelerate.
        accelerator.load_state(str(resume_path))
        assert_optimizer_binding(optimizer, unet, controlnet)
        logger.info(f"Resumed step {global_step}, epoch {first_epoch}, next batch {first_batch}")

    evaluation_queue = None
    if args.evaluation_mode == "async" and accelerator.is_main_process:
        evaluation_queue = AsyncEvaluationQueue(
            output_dir, args, accelerator.unwrap_model(unet), run_config, dataset_split,
        )
        if (resume_path and global_step) or args.validate_initialization:
            evaluation_queue.submit(accelerator.unwrap_model(unet), accelerator.unwrap_model(controlnet),
                                    global_step, final=global_step >= args.max_train_steps)
        logger.info(f"Evaluation worker started on physical GPU {args.evaluation_gpu}; unlabeled forecasts run only at the final step.")
    elif args.validate_initialization and accelerator.is_main_process and not resume_path:
        cuda_devices = [accelerator.device.index] if accelerator.device.type == "cuda" else []
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng(devices=cuda_devices):
                log_validation(vae, text_encoder, tokenizer, unet, controlnet, noise_scheduler,
                               args, accelerator, weight_dtype, global_step, val_dataloader, loss_fn_lpips)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    image_logs = None
    training_complete = global_step >= args.max_train_steps
    stopped = False
    training_started = time.perf_counter()
    with StopAtUpdateBoundary() as stop_request:
        for epoch in range(first_epoch, args.num_train_epochs):
            if training_complete:
                break
            train_loader_generator.manual_seed(args.seed + epoch)
            offset = first_batch if epoch == first_epoch else 0
            epoch_dataloader = accelerator.skip_first_batches(train_dataloader, offset) if offset else train_dataloader
            previous_batch_end = time.perf_counter()
            for step, batch in enumerate(epoch_dataloader, start=offset):
                batch_received = time.perf_counter()
                input_seconds = batch_received - previous_batch_end
                with accumulate_joint_models(accelerator, controlnet, unet):
                    # Convert images to latent space
                    with torch.no_grad():
                        latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = encode_text_conditioning(text_encoder, batch["input_ids"])

                    controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                    if args.temporal:


                        controlnet_image = einops.rearrange(controlnet_image, 'b t c h w -> b c t h w')


                    if args.temporal and hasattr(args, 'drop_temporal_cond') and args.drop_temporal_cond > 0:
                        if random.random() < args.drop_temporal_cond:

                            temporal_dim = controlnet_image.shape[2]


                            min_keep = 1
                            max_keep = temporal_dim
                            num_keep = random.randint(min_keep, max_keep)


                            keep_indices = sorted(random.sample(range(temporal_dim), num_keep))


                            mask = torch.zeros_like(controlnet_image)
                            mask[:, :, keep_indices, :, :] = 1.0
                            controlnet_image = controlnet_image * mask


                            if args.num_metadata > 0 and "conditioning_metadata" in batch:
                                cond_metadata_temp = batch["conditioning_metadata"]

                                mask_metadata = torch.zeros_like(cond_metadata_temp)
                                mask_metadata[:, :, keep_indices] = 1.0
                                batch["conditioning_metadata"] = cond_metadata_temp * mask_metadata

                            logger.debug(f"Temporal dropout: keeping {num_keep}/{temporal_dim} time steps: {keep_indices}")

                    if args.controlnet_scales is not None:

                        temporal_dim = controlnet_image.shape[2] # T in (B, C, T, H, W)
                        if len(args.controlnet_scales) == temporal_dim:

                            scales = torch.tensor(args.controlnet_scales, device=controlnet_image.device, dtype=controlnet_image.dtype)
                            scales = scales.view(1, 1, -1, 1, 1)


                            controlnet_image = controlnet_image * scales
                        else:
                            logger.warning(
                                f"Expected {temporal_dim} temporal scales, got {len(args.controlnet_scales)}; scales were not applied."
                            )
                    metadata = None
                    cond_metadata = None
                    if args.num_metadata > 0:
                        metadata = batch["metadata"]  # (N, num_md)

                        keep_mask = torch.rand_like(metadata) > args.metadata_dropout
                        metadata = metadata * keep_mask + 0. * ~keep_mask * torch.ones_like(metadata)  # set drop metadata to -1

                        cond_metadata = batch["conditioning_metadata"]
                        keep_mask = torch.rand_like(cond_metadata) > args.metadata_dropout
                        cond_metadata = cond_metadata * keep_mask + 0. * ~keep_mask * torch.ones_like(cond_metadata)

                    if args.num_cond > 1:
                        nc = args.num_channels
                        if random.random() < args.drop_cond:
                            num_drop = random.choice(list(range(args.num_cond - 1)))
                            drop_idxs = random.sample(list(range(args.num_cond)), k=num_drop)
                            for idx in drop_idxs:
                                controlnet_image[:, nc*idx:nc*(idx+1), :, :] = torch.zeros_like(controlnet_image[:, nc*idx:nc*(idx+1), :, :])
                                if args.num_metadata > 0:
                                    cond_metadata[:, :, idx] = torch.zeros_like(cond_metadata[:, :, idx])

                    if args.multi:
                        cond_metadata = einops.rearrange(cond_metadata, 'b m t -> t b m')
                        cond_metadata = [cond_md for cond_md in cond_metadata]

                    controlnet_scale = control_scale(args)

                    down_block_res_samples, mid_block_res_sample = controlnet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        controlnet_cond=controlnet_image,
                        conditioning_scale=controlnet_scale,
                        metadata=metadata,
                        cond_metadata=cond_metadata,
                        return_dict=False,
                        **target_metadata_kwargs(controlnet, unet, metadata),
                    )

                    # Predict the noise residual
                    model_pred = unet(
                        noisy_latents,
                        timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                        metadata=metadata,
                        down_block_additional_residuals=[
                            sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                        ],
                        mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    ).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")


                    # The selected preset combines direct denoising MSE with
                    # image-space MSE and LPIPS; each weight remains configurable.
                    noise_loss = F.mse_loss(model_pred.float(), target.float())
                    clean_latents = predicted_clean_latents(noisy_latents, model_pred, timesteps, noise_scheduler)
                    predicted_images, image_mse, perceptual_loss = image_objectives(
                        vae, clean_latents, batch["pixel_values"], loss_fn_lpips, args.lpips_loss_weight,
                    )
                    # Forward values use the approved hard masks. The area
                    # surrogate sends gradients only through decoded predictions.
                    loss_area = predicted_images.new_zeros(())
                    if args.morph_loss_lambda > 0:
                        loss_area = hard_foreground_area_loss(
                            predicted_images, batch["pixel_values"], temperature=args.morph_temperature,
                            target_mask=batch.get("target_foreground_mask"),
                        )
                    loss = (
                        args.image_mse_weight * image_mse
                        + args.lpips_loss_weight * perceptual_loss
                        + args.diffusion_loss_weight * noise_loss
                        + args.morph_loss_lambda * loss_area
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite training loss before optimizer step {global_step + 1}")
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        params_to_clip = list(trainable_parameters(unet, controlnet))
                        grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                        if not torch.isfinite(grad_norm):
                            raise FloatingPointError(f"Non-finite gradient norm before optimizer step {global_step + 1}")
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=args.set_grads_to_none)
                compute_seconds = time.perf_counter() - batch_received

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    #################################################################
                    # CUSTOM: Log Wandb
                    if args.wandb is not None and accelerator.is_main_process:
                        wandb.log(
                            {"Train/loss": loss.detach().item(), "Train/lr": lr_scheduler.get_last_lr()[0], "Train/step": global_step}
                        )
                    #################################################################

                    progress_bar.update(1)
                    global_step += 1

                    if accelerator.is_main_process:
                        # if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        if args.validation_steps > 0 and global_step > 0 and global_step % args.validation_steps == 0 :
                            if evaluation_queue is not None:
                                evaluation_queue.submit(accelerator.unwrap_model(unet), accelerator.unwrap_model(controlnet),
                                                        global_step, final=global_step >= args.max_train_steps)
                            else:
                                with accelerator.autocast():
                                    image_logs = log_validation(
                                        vae, text_encoder, tokenizer, unet, controlnet, noise_scheduler,
                                        args, accelerator, weight_dtype, global_step, val_dataloader,
                                        loss_fn_lpips, inference_dataloader=None,
                                    )

                if accelerator.sync_gradients:
                    logs = {
                        "loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm.detach().item(),
                        "metadata_lr": lr_scheduler.get_last_lr()[1],
                        "image_mse": image_mse.detach().item(), "lpips": perceptual_loss.detach().item(),
                        "diffusion_mse": noise_loss.detach().item(), "hard_otsu_area": loss_area.detach().item(),
                        "hard_otsu_area_fraction": loss_area.detach().item() / (predicted_images.shape[-2] * predicted_images.shape[-1]),
                        "weighted_area_loss": args.morph_loss_lambda * loss_area.detach().item(),
                        "input_seconds": input_seconds, "compute_seconds": compute_seconds,
                        "elapsed_seconds": time.perf_counter() - training_started,
                    }
                    progress_bar.set_postfix(**logs)
                    accelerator.log(logs, step=global_step)

                    if accelerator.is_main_process:

                        train_metrics_file_path = os.path.join(args.output_dir, "training_metrics.txt")


                        with open(train_metrics_file_path, "a") as f:

                            f.write(
                                f"Step: {global_step}, Loss: {logs['loss']:.4f}, LR: {logs['lr']}, "
                                f"ImageMSE: {logs['image_mse']:.6f}, LPIPS: {logs['lpips']:.6f}, "
                                f"DiffusionMSE: {logs['diffusion_mse']:.6f}, HardOtsuArea: {logs['hard_otsu_area']:.2f}, "
                                f"WeightedArea: {logs['weighted_area_loss']:.6f}, GradNorm: {logs['grad_norm']:.6f}\n"
                            )
                        with (output_dir / "training_metrics.jsonl").open("a") as stream:
                            stream.write(json.dumps(dict(step=global_step, **logs)) + "\n")


                    next_batch = step + 1
                    training_state.update(
                        global_step=global_step,
                        epoch=epoch + (next_batch == len(train_dataloader)),
                        next_batch=0 if next_batch == len(train_dataloader) else next_batch,
                    )
                    stopped = stop_request.across_processes(accelerator)
                    evaluation_error = evaluation_queue.error() if evaluation_queue is not None else None
                    regular_save = global_step % args.checkpointing_steps == 0
                    preempt_save = args.checkpoint_preempt_steps is not None and global_step % args.checkpoint_preempt_steps == 0
                    if regular_save or preempt_save or stopped or evaluation_error or global_step >= args.max_train_steps:
                        save_path = output_dir / f"checkpoint-{global_step}"
                        save_training_checkpoint(accelerator, save_path, training_state, dataset_split)
                        logger.info(f"Saved complete joint training state to {save_path}")
                        if accelerator.is_main_process and preempt_save and not regular_save:
                            for previous in output_dir.glob("checkpoint-*"):
                                match = re.fullmatch(r"checkpoint-(\d+)", previous.name)
                                if match and (previous / "checkpoint_complete.json").is_file():
                                    previous_step = int(match.group(1))
                                    if previous_step < global_step and previous_step % args.checkpointing_steps != 0:
                                        shutil.rmtree(previous)
                    if evaluation_error:
                        evaluation_queue.close("failed")
                        raise RuntimeError(f"Evaluation failed; training checkpoint saved: {evaluation_error}")
                    if stopped:
                        break

                if global_step >= args.max_train_steps:
                    training_complete = True
                    break
                previous_batch_end = time.perf_counter()
            if training_complete or stopped:
                logger.info("Training stopped at a completed optimizer update.")

                break
    progress_bar.close()
    if stopped:
        logger.info("Interrupted run saved; use --resume_from_checkpoint to continue it.")
        if evaluation_queue is not None:
            evaluation_queue.close("paused")
        accelerator.end_training()
        return

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()


    if accelerator.is_main_process:

        controlnet = accelerator.unwrap_model(controlnet)
        unet = accelerator.unwrap_model(unet)
        if evaluation_queue is not None:
            evaluation_queue.submit(unet, controlnet, global_step, final=True)
            evaluation_queue.close("completed")
        if args.skip_final_pipeline_export:
            logger.info("Complete final checkpoint retained; skipping duplicate pipeline exports.")
        else:
            export_path = os.path.join(args.output_dir, "unet_final")
            unet.save_pretrained(export_path)
            controlnet.save_pretrained(args.output_dir)
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                args.pretrained_model_name_or_path, text_encoder=text_encoder,
                vae=vae, unet=unet, controlnet=controlnet,
            )
            pipeline.save_pretrained(args.output_dir)
            logger.info(f"Successfully saved the full inference pipeline to: {args.output_dir}")


    if accelerator.is_main_process:
        if evaluation_queue is not None:
            logger.info("Training complete. GPU 1 will finish final validation, unlabeled forecasts and MC Dropout; see evaluation_status.json.")
        elif args.inference_data_dir or args.validation_steps > 0:
            # Periodic validation excludes unlabeled cells; forecast them once.
            if args.inference_data_dir or global_step % args.validation_steps != 0:
                log_validation(vae, text_encoder, tokenizer, unet, controlnet, noise_scheduler,
                               args, accelerator, weight_dtype, global_step, val_dataloader, loss_fn_lpips,
                               inference_dataloader=inference_dataloader)

    if args.run_uncertainty_analysis and accelerator.is_main_process and evaluation_queue is None:
            logger.info("***** Starting Uncertainty Analysis on Validation Set *****")

            # Run final visualization with unwrapped models on the main process.
            unwrapped_controlnet = accelerator.unwrap_model(controlnet)
            unet = accelerator.unwrap_model(unet)


            if accelerator.is_main_process:
                uncertainty_dir = Path(args.output_dir) / args.uncertainty_output_dir
                uncertainty_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Uncertainty maps will be saved to: {uncertainty_dir}")

            # This validation loader is not sharded; avoid duplicate predictions.
            for batch in tqdm(val_dataloader, desc="Uncertainty Analysis", disable=not accelerator.is_main_process):
                if batch['pixel_values'].shape[0] == 0:
                    continue


                with accelerator.autocast():
                    generate_with_uncertainty(
                        vae=vae,
                        text_encoder=text_encoder,
                        unet=unet,
                        controlnet=unwrapped_controlnet,
                        noise_scheduler=noise_scheduler,
                        accelerator=accelerator,
                        weight_dtype=weight_dtype,
                        args=args,
                        data_batch=batch,
                        output_dir=Path(args.output_dir) / args.uncertainty_output_dir,
                        num_mc_samples=args.mc_samples
                    )


    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
